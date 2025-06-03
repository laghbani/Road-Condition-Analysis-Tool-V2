import hashlib
import json
from pathlib import Path
from datetime import datetime

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from iso_weighting import calc_awv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from progress_ui import ProgressWindow
import cv2
from sensor_msgs.msg import PointCloud2, Image, CompressedImage
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
from videopc_widget import _pc_to_xyz


CITY_BBOX = {
    "Merseburg": ((51.34, 51.38), (11.95, 12.07)),
    "Halle": ((51.45, 51.55), (11.85, 12.05)),
    "Leipzig": ((51.28, 51.42), (12.20, 12.48)),
    "Berlin": ((52.30, 52.65), (13.00, 13.80)),
    "Munich": ((48.09, 48.23), (11.45, 11.65)),
}


def guess_city(lat: float, lon: float) -> str:
    for name, ((la0, la1), (lo0, lo1)) in CITY_BBOX.items():
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return name
    return f"{lat:.2f}_{lon:.2f}"


def c_for(v: float) -> str:
    if v < 1.72:
        return "green"
    elif v < 2.12:
        return "yellow"
    elif v < 2.54:
        return "orange"
    elif v < 3.19:
        return "red"
    else:
        return "purple"


def save_plot(df: pd.DataFrame, cols: list[str], path: Path, title: str) -> None:
    plt.figure()
    for col in cols:
        if col not in df.columns:
            continue
        plt.plot(df["time"], df[col], label=col)
    plt.xlabel("time [s]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def prepare_gps(gps_df: pd.DataFrame, df: pd.DataFrame, comfort: bool,
                peak_height: float, peak_dist: float, max_peak: bool) -> pd.DataFrame:
    if df.empty:
        return gps_df.copy()
    fs = 1.0 / np.median(np.diff(df["time"])) if len(df) > 1 else 100.0
    res = calc_awv(
        df["accel_corr_x"], df["accel_corr_y"], df["accel_corr_z"], fs,
        comfort=comfort,
        peak_height=peak_height,
        peak_dist=peak_dist,
        max_peak=max_peak,
    )
    gps = gps_df.copy()
    gps["awv"] = np.interp(gps["time"], df["time_abs"], res["awv"])
    gps["color"] = gps["awv"].apply(c_for)
    gps["peak"] = False
    if len(res["peaks"]):
        peak_times = df.loc[res["peaks"], "time_abs"].to_numpy()
        tol = 0.1
        gps["peak"] = gps["time"].apply(lambda t: bool(np.any(np.abs(t - peak_times) <= tol)))
    return gps


def save_map(gps: pd.DataFrame, path: Path) -> None:
    try:
        import folium
    except Exception:
        return
    if gps.empty:
        return
    lat0 = gps["lat"].mean()
    lon0 = gps["lon"].mean()
    fmap = folium.Map(location=[lat0, lon0], zoom_start=16, max_zoom=30, min_zoom=10)
    for row in gps.itertuples():
        if row.peak:
            folium.Marker(
                location=[row.lat, row.lon],
                icon=folium.Icon(color="black", icon="star"),
                popup=f"Peak@{row.time:.2f}"
            ).add_to(fmap)
        else:
            folium.CircleMarker(
                location=[row.lat, row.lon], radius=4,
                color=row.color, fill=True
            ).add_to(fmap)
    fmap.save(str(path))




def rot_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return minimal rotation matrix mapping `v1 to v2.

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
    """Add `speed_mps column to *work* derived from gps_df."""
    if gps_df is None or len(gps_df) < 3:
        work["speed_mps"] = np.nan
        return work

    # ➊ NICHT doppelt führen – falls die Spalte bereits existiert, entfernen
    work = work.drop(columns=["speed_mps"], errors="ignore")

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

    # ➋ Eine konsolidierte Spalte herstellen – robust gegen Duplikate
    if {"speed_mps_x", "speed_mps_y"}.issubset(out.columns):
        base = out.pop("speed_mps_x")
        gps = out.pop("speed_mps_y")
        out["speed_mps"] = gps.combine_first(base)
    else:
        out = out.rename(columns={"speed_mps": "speed_mps"})
    out["speed_mps"] = out["speed_mps"].interpolate(limit_direction="both")

    return out


G_STD = 9.80665
FSR_2G = 2 * G_STD
FSR_8G = 8 * G_STD
CAND_G = np.array([2, 4, 8, 16]) * G_STD        #  19.6, 39.2 … 156.9 m/s²


def _get_qt_widget(obj, name: str):
    """Return a Qt widget class from the same binding as *obj*."""
    pkg = obj.__class__.__module__.split('.')[0]
    try:
        mod = __import__(f"{pkg}.QtWidgets", fromlist=[name])
        return getattr(mod, name)
    except Exception:
        pass
    for pkg in ("PyQt5", "PySide6"):
        try:
            mod = __import__(f"{pkg}.QtWidgets", fromlist=[name])
            return getattr(mod, name)
        except Exception:
            continue
    return None


def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _img_to_bgr(obj):
    if isinstance(obj, Image):
        return CvBridge().imgmsg_to_cv2(obj, desired_encoding="bgr8")
    if isinstance(obj, (bytes, bytearray, memoryview)):
        arr = cv2.imdecode(np.frombuffer(obj, np.uint8), cv2.IMREAD_COLOR)
        return arr
    return np.asarray(obj)


def write_gpx(df: pd.DataFrame, path: Path) -> None:
    """Write GPS data *df* to a very small GPX file."""
    with open(path, "w") as f:
        f.write('<gpx version="1.1">\n<trk><trkseg>\n')
        for _, row in df.iterrows():
            t = datetime.utcfromtimestamp(float(row["time"])).isoformat() + "Z"
            f.write(f'<trkpt lat="{row.lat}" lon="{row.lon}"><ele>{row.alt}</ele><time>{t}</time></trkpt>\n')
        f.write('</trkseg></trk></gpx>')


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
    mag = np.linalg.norm(df[["accel_x", "accel_y", "accel_z"]].to_numpy(), axis=1)
    ser = pd.Series(mag)
    std = ser.rolling(win, center=True).std()
    mean = ser.rolling(win, center=True).mean()
    mask = (std < thr_sigma) & (np.abs(mean - G_STD) < 0.2)
    if mask.any():
        idx = mask.idxmax()
        j0 = max(0, idx - win // 2)
        j1 = min(len(df), idx + win // 2 + 1)
        return df.iloc[j0:j1][["accel_x", "accel_y", "accel_z"]].mean().to_numpy()
    return None


def gravity_from_quat(df: pd.DataFrame) -> np.ndarray:
    q = df[["ox", "oy", "oz", "ow"]].to_numpy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rots = R.from_quat(q)
    g_world = np.array([0.0, 0.0, G_STD])
    return rots.inv().apply(g_world)


def remove_gravity_lowpass(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate gravity via low-pass filter and remove it from acceleration."""
    bias_vec = find_stationary_bias(df)
    if bias_vec is None:
        bias_vec = np.zeros(3)
    acc_bias = df[["accel_x", "accel_y", "accel_z"]].to_numpy() - bias_vec

    dt = np.median(np.diff(df["time"]))
    fs = 1.0 / dt if dt > 0 else 100.0
    fc = 0.15
    b, a = butter(2, fc / (0.5 * fs), "low")
    g_est = filtfilt(b, a, acc_bias, axis=0)

    scale = G_STD / np.median(np.linalg.norm(g_est, axis=1))
    g_est *= scale

    acc_corr = acc_bias - g_est
    return acc_corr, g_est, bias_vec


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
    """Calculate 3x3 rotation matrix sensor→vehicle or return `None."""

    # --- Z-axis via gravity vector -------------------------------------
    if {"grav_x", "grav_y", "grav_z"}.issubset(df.columns):
        z_sens = df[["grav_x", "grav_y", "grav_z"]].mean().to_numpy()
    elif {"accel_x", "accel_y", "accel_z"}.issubset(df.columns):
        g_sens = df[["accel_x", "accel_y", "accel_z"]].mean().to_numpy()
        if np.linalg.norm(g_sens) < 1:
            return None
        z_sens = g_sens / np.linalg.norm(g_sens)
    else:
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

    QFileDialog = _get_qt_widget(self, "QFileDialog")
    if QFileDialog is not None:
        start_dir = "/home/afius/Desktop/anomaly-data-hs-merseburg"
        sel = QFileDialog.getExistingDirectory(self, "Select export directory", start_dir)
        if not sel:
            return
        root = Path(sel)
    else:
        root = Path("/home/afius/Desktop/anomaly-data-hs-merseburg")

    # Zielordner automatisch erzeugen
    label_date = datetime.now().strftime("%Y%m%d")
    city = "unknown"
    if gps_df is not None and not gps_df.empty:
        city = guess_city(float(gps_df['lat'].iat[0]), float(gps_df['lon'].iat[0]))
    dest = root / f"{label_date}_{city}_{bag_root.stem}"
    dest.mkdir(parents=True, exist_ok=True)
    self.last_export_dir = str(dest)   # für den GUI-Check
    exporter_sha = sha1_of_file(Path(__file__))

    progress = ProgressWindow("Export", ["Schreibe CSV …"], parent=self)
    progress.set_bar_range(len(self.dfs))

    for i, (topic, df) in enumerate(self.dfs.items()):
        progress.set_bar_value(i)
        if progress.wasCanceled():
            break
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
                work["gyro_x"] = gyro[:, 0]
                work["gyro_y"] = gyro[:, 1]
                work["gyro_z"] = gyro[:, 2]

            # Speed aus GPS-Daten ableiten
            work = add_speed(work, gps_df)

            abs_a = np.linalg.norm(df[["accel_x", "accel_y", "accel_z"]].to_numpy(), axis=1)
            fsr = detect_fsr(abs_a)
            clipped = (df[["accel_x", "accel_y", "accel_z"]].abs() >= 0.98 * fsr).any(axis=1)

            if has_quat:
                g_vec = gravity_from_quat(
                    pd.DataFrame(ori, columns=["ox", "oy", "oz", "ow"]))
                acc_corr = df[["accel_x", "accel_y", "accel_z"]].to_numpy() - g_vec
                g_est = g_vec
                bias_vec = None
                comp_type = "quaternion"
            else:
                acc_corr, g_est, bias_vec = remove_gravity_lowpass(df)
                comp_type = "lowpass_bias"

            work["accel_corr_x"], work["accel_corr_y"], work["accel_corr_z"] = acc_corr.T
            work["grav_x"], work["grav_y"], work["grav_z"] = g_est.T

            # Rotation & Beschleunigung transformieren
            rot_mat = self._resolve_rotation(topic, work)
            rot_avail = rot_mat is not None
            if rot_avail:
                R = np.asarray(rot_mat)
                veh = acc_corr @ R.T
                work["accel_veh_x"], work["accel_veh_y"], work["accel_veh_z"] = veh.T

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

            self.dfs[topic] = work

            header = {
                "file_format": "imu_v1",
                "sensor_topic": topic,
                "sensor_model": get_sensor_model(bag_root, topic),
                "coordinate_frame": first_frame_id(samps),
                "sensor_fs_accel_g": round(fsr / G_STD, 3),
                "bias_vector_mps2": [round(x, 3) for x in bias_vec] if bias_vec is not None else None,
                "g_compensation": comp_type,
                "vehicle_rot_mat": rot_mat.tolist() if rot_mat is not None else None,
                "rotation_available": rot_avail,
                "sampling_rate_hz": round(fs, 2),
                "exporter_sha1": exporter_sha,
            }

            cols = ["time", "time_abs", "accel_x", "accel_y", "accel_z", "speed_mps"]
            if has_gyro:
                cols += ["gyro_x", "gyro_y", "gyro_z"]
            cols += ["accel_corr_x", "accel_corr_y", "accel_corr_z",
                     "grav_x", "grav_y", "grav_z"]
            if rot_mat is not None:
                cols += ["accel_veh_x", "accel_veh_y", "accel_veh_z"]
            cols += ["|a|_raw", "|ω|_raw", "clipped_flag", "label_id", "label_name"]

            out_df = work[cols]

            stem = f"{topic.strip('/').replace('/', '__')}_{bag_root.stem}__imu_v1"
            csv_path = dest / f"{stem}.csv"
            meta_path = dest / f"{stem}.json"

            # --- DEBUG: prüfen, ob g sauber entfernt wurde -------------------
            vec_mean = acc_corr.mean(axis=0)
            rms_corr = np.sqrt((acc_corr ** 2).mean())
            print(
                f"[{topic}]  μ(accel_corr_x, accel_corr_y, accel_corr_z) = {vec_mean.round(2)}  "
                f" RMS(|a_corr|) = {rms_corr:.2f}"
            )

            def _conv(o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

            meta_path.write_text(json.dumps(header, indent=2, default=_conv))
            out_df.to_csv(csv_path, index=False)

            detail_dir = dest / stem
            detail_dir.mkdir(exist_ok=True)

            save_plot(work, ["accel_x", "accel_y", "accel_z"], detail_dir / "acc_raw.png", "Raw Acceleration")
            save_plot(work, ["grav_x", "grav_y", "grav_z"], detail_dir / "gravity.png", "Gravity Vector")
            if {"accel_veh_x", "accel_veh_y", "accel_veh_z"}.issubset(work.columns):
                save_plot(work, ["accel_veh_x", "accel_veh_y", "accel_veh_z"], detail_dir / "acc_vehicle.png", "Vehicle Frame")
            save_plot(work, ["awx", "awy", "awz"], detail_dir / "weighted.png", "Weighted")
            save_plot(work, ["rms_x", "rms_y", "rms_z"], detail_dir / "rms.png", "Running RMS")

        except Exception as exc:
            QMessageBox = _get_qt_widget(self, "QMessageBox")
            if QMessageBox is None:
                print(f"{topic}: {exc}")
            else:
                QMessageBox.critical(self, "Export-Fehler", f"{topic}: {exc}", QMessageBox.Ok)
            continue
        

    if gps_df is not None and not gps_df.empty:
        topic0 = next(iter(self.dfs))
        df0 = self.dfs[topic0]
        gps_comfort = prepare_gps(gps_df, df0, True,
                                  self.peak_threshold, self.peak_distance, self.use_max_peak)
        gps_health = prepare_gps(gps_df, df0, False,
                                 self.peak_threshold, self.peak_distance, self.use_max_peak)
        track = gps_df.copy()
        track[["awv_comfort", "color_comfort", "peak_comfort"]] = gps_comfort[["awv", "color", "peak"]]
        track[["awv_health", "color_health", "peak_health"]] = gps_health[["awv", "color", "peak"]]
        track_path = dest / f"{bag_root.stem}_track.csv"
        track.to_csv(track_path, index=False)
        write_gpx(gps_df, dest / f"{bag_root.stem}_track.gpx")
        save_map(gps_comfort, dest / f"{bag_root.stem}_comfort.html")
        save_map(gps_health, dest / f"{bag_root.stem}_health.html")

    # --- export peak media ----------------------------------------------
    if hasattr(self, "iso_metrics") and self.iso_metrics:
        topic0 = next(iter(self.dfs))
        peaks = self.iso_metrics.get(topic0, {}).get("peaks", [])
        if len(peaks):
            peak_times = self.dfs[topic0].loc[peaks, "time_abs"].to_numpy()
            media_dir = dest / "peaks"
            media_dir.mkdir(exist_ok=True)
            pre = getattr(self.tab_vpc, "spn_pre", None)
            post = getattr(self.tab_vpc, "spn_post", None)
            pre = pre.value() if pre else 2.0
            post = post.value() if post else 2.0
            t0 = self.t0 or 0.0
            for pt in peak_times:
                pdir = media_dir / f"{pt - t0:.2f}"
                pdir.mkdir(exist_ok=True)
                for vtopic, frames in self.video_frames_by_topic.items():
                    times = self.video_times_by_topic.get(vtopic, [])
                    tr = np.array(times) - t0
                    start = pt - t0 - pre
                    end = pt - t0 + post
                    i0 = int(np.searchsorted(tr, start, "left"))
                    i1 = int(np.searchsorted(tr, end, "right"))
                    for j, fr in enumerate(frames[i0:i1]):
                        img = _img_to_bgr(fr)
                        img_path = pdir / f"{vtopic.strip('/').replace('/', '__')}_{j:03d}.png"
                        if img is not None:
                            cv2.imwrite(str(img_path), img)
                for ptopic, frames in self.pc_frames_by_topic.items():
                    times = self.pc_times_by_topic.get(ptopic, [])
                    tr = np.array(times) - t0
                    start = pt - t0 - pre
                    end = pt - t0 + post
                    i0 = int(np.searchsorted(tr, start, "left"))
                    i1 = int(np.searchsorted(tr, end, "right"))
                    for j, pc in enumerate(frames[i0:i1]):
                        pts = _pc_to_xyz(pc)
                        pc_path = pdir / f"{ptopic.strip('/').replace('/', '__')}_{j:03d}.txt"
                        np.savetxt(pc_path, pts, fmt="%.3f")

    progress.set_bar_value(len(self.dfs))
    progress.accept()

    if hasattr(self, "statusBar"):
        self.statusBar().showMessage(f"Export abgeschlossen → {dest}")

    QMessageBox = _get_qt_widget(self, "QMessageBox")
    if QMessageBox is None:
        print(f"CSV + JSON liegen in: {dest}")
    else:
        QMessageBox.information(self, "Export fertig",
                                f"CSV + JSON liegen in:\n{dest}")
