#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU Labeling Tool – v0.6.1
===========================
*Stand: 30-Mai-2025*

v0.6.1 – Fix
------------
* Fallback-Mechanismus für Matplotlib-Styles, damit auch sehr alte
  Matplotlib-Versionen (ohne `matplotlib.style) funktionieren.
"""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import json
import os

# ---------------------------------------------------------------------------#
# Matplotlib – robuster Style-Loader                                         #
# ---------------------------------------------------------------------------#
try:
    import matplotlib
    matplotlib.use("Qt5Agg") 
    import matplotlib.pyplot as _plt

    # kleiner Helfer: Style sicher anwenden
    def _set_mpl_style() -> None:
        for sty in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid"):
            try:
                _plt.style.use(sty)           # neuer Weg
                break
            except Exception:
                try:
                    matplotlib.style.use(sty)  # alter Weg
                    break
                except Exception:
                    continue
        # minimale Basis-Parameter (werden immer gesetzt)
        matplotlib.rcParams.update({
            "figure.autolayout": False,
            "axes.titlesize": "medium",
            "axes.labelsize": "small",
            "legend.fontsize": "small",
            "lines.linewidth": 1.0,
        })

    _set_mpl_style()

    # Canvas-Backend erst nach Style laden
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:   # nur falls wirklich Qt6-Umgebung aktiv ist
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.widgets import SpanSelector
except ImportError as exc:
    print("[FATAL] Matplotlib-Import fehlgeschlagen:", exc)
    sys.exit(1)

# ---------------------------------------------------------------------------#
# Qt-Import (PySide 6 bevorzugt, sonst PyQt 5)                                #
# ---------------------------------------------------------------------------#
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QFileDialog, QMessageBox,
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QAction, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    )
    from PyQt5.QtCore import Qt
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QFileDialog, QMessageBox,
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QAction, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
    )
    from PySide6.QtCore import Qt

# ---------------------------------------------------------------------------#
# ROS-Import                                                                 #
# ---------------------------------------------------------------------------#
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu, NavSatFix
    from imu_csv_export_v2 import (
        export_csv_smart_v2,
        remove_gravity_lowpass,
        auto_vehicle_frame,
        gravity_from_quat,
    )
except ModuleNotFoundError:
    print("[FATAL] ROS 2-Python-Pakete nicht gefunden. Bitte ROS 2 installieren & sourcen.")
    sys.exit(1)

# ===========================================================================
# Label-Mapping
# ===========================================================================
ANOMALY_TYPES: Dict[str, Dict[str, str | int]] = {
    "normal": {"score": 0,  "color": "#00FF00"},
    "depression": {"score": 4, "color": "#FF0000"},
    "cover": {"score": 2, "color": "#FFA500"},
    "cobble road/ traditional road": {"score": 1, "color": "#FFFF00"},
    "transverse grove": {"score": 1, "color": "#00FF00"},
    "gravel road": {"score": 4, "color": "#FAF2A1"},
    "cracked / irregular pavement and aspahlt": {"score": 2, "color": "#E06D06"},
    "bump": {"score": 1, "color": "#54F2F2"},
    "uneven/repaired asphalt road": {"score": 1, "color": "#A30B37"},
    "Damaged pavemant / asphalt road": {"score": 4, "color": "#2B15AA"},
}
UNKNOWN_ID, UNKNOWN_NAME, UNKNOWN_COLOR = 99, "unknown", "#808080"

# ===========================================================================
# Dataclass IMU
# ===========================================================================
@dataclass
class ImuSample:
    t: int
    msg: Imu

    @property
    def time_abs(self) -> float:
        return self.t / 1e9

    @property
    def lin_acc(self) -> Tuple[float, float, float]:
        la = self.msg.linear_acceleration
        return la.x, la.y, la.z


# ===========================================================================
# Topic-Dialog
# ===========================================================================
class TopicDialog(QDialog):
    def __init__(self, topics: List[str], active: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("IMU Settings")
        self.resize(380, 420)
        v = QVBoxLayout(self)

        v.addWidget(QLabel("☑ anzeigen    •  Drag-&-Drop = Reihenfolge",
                           alignment=Qt.AlignCenter))

        self.listw = QListWidget()
        self.listw.setSelectionMode(QListWidget.NoSelection)
        self.listw.setDragDropMode(QListWidget.InternalMove)
        for t in topics:
            it = QListWidgetItem(t)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            it.setCheckState(Qt.Checked if t in active else Qt.Unchecked)
            self.listw.addItem(it)
        v.addWidget(self.listw)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def selected_topics(self) -> List[str]:
        sel: List[str] = []
        for i in range(self.listw.count()):
            it = self.listw.item(i)
            if it.checkState() == Qt.Checked:
                sel.append(it.text())
        return sel


# ===========================================================================
# Main-Window
# ===========================================================================
class MainWindow(QMainWindow):
    LABEL_IDS = {name: i + 1 for i, name in enumerate(ANOMALY_TYPES)}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROS 2 IMU-Labeling Tool")
        self.resize(1500, 900)

        # Daten-Strukturen
        self.bag_path: pathlib.Path | None = None
        self.samples: dict[str, list[ImuSample]] = {}
        self.dfs: dict[str, pd.DataFrame] = {}
        self.t0: float | None = None
        self._gps_df: pd.DataFrame | None = None

        # Runtime-State
        self.active_topics: list[str] = []
        self.ax_topic: dict[object, str] = {}
        self.span_selector: dict[str, SpanSelector] = {}
        self.current_span: dict[str, Tuple[float, float]] = {}

        self._build_menu()
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._mouse_press)

        hl = QHBoxLayout()
        vbox.addLayout(hl)
        hl.addWidget(QLabel("Segment Label:"))
        self.cmb = QComboBox()
        for n in ANOMALY_TYPES:
            self.cmb.addItem(n, userData=n)
        hl.addWidget(self.cmb)
        hl.addStretch()

    # ------------------------------------------------------------------ Menu
    def _build_menu(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("&Datei")
        act_open = QAction("&Bag öffnen …", self)
        act_open.triggered.connect(self._open_bag)
        m_file.addAction(act_open)

        self.act_export = QAction("&CSV exportieren", self)
        self.act_export.setEnabled(False)
        self.act_export.triggered.connect(
            lambda: export_csv_smart_v2(self, gps_df=self._gps_df))
        m_file.addAction(self.act_export)

        m_file.addSeparator()
        m_file.addAction("Beenden", lambda: QApplication.instance().quit())

        m_imu = mb.addMenu("&IMU Settings")
        self.act_topics = QAction("Topics auswählen …", self)
        self.act_topics.setEnabled(False)
        self.act_topics.triggered.connect(self._configure_topics)
        m_imu.addAction(self.act_topics)

        m_view = mb.addMenu("&View")
        self.act_verify = QAction("Verify your labeling", self)
        self.act_verify.setEnabled(False)
        self.act_verify.triggered.connect(lambda: self._draw_plots(verify=True))
        m_view.addAction(self.act_verify)

        self.act_show_raw = QAction("Show raw acceleration", self, checkable=True)
        self.act_show_raw.setChecked(True)
        self.act_show_raw.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_raw)

        self.act_show_corr = QAction("Show g-corrected acceleration", self, checkable=True)
        self.act_show_corr.setChecked(True)
        self.act_show_corr.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_corr)

        self.act_show_veh = QAction("Show vehicle-frame acceleration", self, checkable=True)
        self.act_show_veh.setChecked(False)
        self.act_show_veh.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_veh)

        act_check = QAction("Export Readiness …", self)
        act_check.setEnabled(False)
        act_check.triggered.connect(self._check_export_status)
        m_view.addAction(act_check)
        self.act_check = act_check

    # ------------------------------------------------------------------ Bag
    def _open_bag(self) -> None:
        pth, _ = QFileDialog.getOpenFileName(
            self, "ROS 2 Bag wählen", "", "ROS 2 Bag (*.db3 *.sqlite3 *.mcap);;Alle Dateien (*)"
        )
        if not pth:
            return
        p = pathlib.Path(pth)
        self.bag_path = p if p.is_file() else p.parent
        self._load_bag()

    def _load_bag(self) -> None:
        assert self.bag_path
        self.samples.clear()
        self.dfs.clear()
        self.t0 = None

        try:
            reader = SequentialReader()
            reader.open(StorageOptions(str(self.bag_path), "sqlite3"), ConverterOptions("cdr", "cdr"))
            topics_info = reader.get_all_topics_and_types()
            imu_topics = [t.name for t in topics_info if t.type == "sensor_msgs/msg/Imu"]
            gps_topic = next((t.name for t in topics_info if t.name == "/lvx_client/navsat"), None)
            for t in imu_topics:
                self.samples[t] = []
        except Exception as exc:
            QMessageBox.critical(self, "Lesefehler", str(exc))
            return

        if not self.samples:
            QMessageBox.information(self, "Keine IMU-Topics", "Es wurden keine IMU-Topics gefunden.")
            return

        # Daten einlesen
        gps_samples: list[tuple] = []
        try:
            while reader.has_next():
                topic, data, ts = reader.read_next()
                if topic in self.samples:
                    self.samples[topic].append(ImuSample(ts, deserialize_message(data, Imu)))
                elif gps_topic and topic == gps_topic:
                    msg = deserialize_message(data, NavSatFix)
                    gps_samples.append((ts / 1e9, msg.latitude, msg.longitude, msg.altitude))
        except Exception as exc:
            QMessageBox.critical(self, "Lesefehler", f"Fehler beim Lesen:\n{exc}")
            return

        self._gps_df = pd.DataFrame(gps_samples, columns=["time", "lat", "lon", "alt"]) if gps_samples else None

        self._build_dfs()
        self._preprocess_all()
        self._set_defaults()
        self._draw_plots()

        self.act_topics.setEnabled(True)
        self.act_verify.setEnabled(True)
        self.act_export.setEnabled(True)
        self.act_check.setEnabled(True)

    # ------------------------------------------------------------------ DataFrame
    def _build_dfs(self) -> None:
        for t, samps in self.samples.items():
            if not samps:
                continue
            ta = np.array([s.time_abs for s in samps])
            if self.t0 is None:
                self.t0 = ta[0]
            tr = ta - self.t0
            df = pd.DataFrame({
                "time": tr, "time_abs": ta,
                "ax": [s.lin_acc[0] for s in samps],
                "ay": [s.lin_acc[1] for s in samps],
                "az": [s.lin_acc[2] for s in samps],
                "label_id": np.full_like(tr, UNKNOWN_ID, int),
                "label_name": [UNKNOWN_NAME] * len(tr),
            })
            self.dfs[t] = df

    # ------------------------------------------------------------------ Preprocess
    def _preprocess_all(self) -> None:
        for topic, df in self.dfs.items():
            samps = self.samples[topic]

            # --- Prüfen, ob Quaternionen verwertbar ---------------------------
            ori = np.array([[s.msg.orientation.x, s.msg.orientation.y,
                            s.msg.orientation.z, s.msg.orientation.w]
                            for s in samps])
            norm_ok = np.abs(np.linalg.norm(ori, axis=1) - 1.0) < 0.05
            var_ok  = np.ptp(ori, axis=0).max() > 1e-3
            has_quat = bool(norm_ok.any() and var_ok)

            # --- g-Kompensation ----------------------------------------------
            if has_quat:
                g_vec = gravity_from_quat(
                    pd.DataFrame(ori, columns=["ox", "oy", "oz", "ow"])
                )
                acc_corr = df[["ax", "ay", "az"]].to_numpy() - g_vec
                g_est = g_vec
            else:
                acc_corr, g_est, _ = remove_gravity_lowpass(df)

            df[["ax_corr", "ay_corr", "az_corr"]] = acc_corr
            df[["g_x", "g_y", "g_z"]] = g_est

            # --- Fahrzeug-Rahmen ---------------------------------------------
            rot = auto_vehicle_frame(df, self._gps_df)
            if rot:
                R = np.array(rot)
                veh = acc_corr @ R.T
                df[["ax_veh", "ay_veh", "az_veh"]] = veh

    # ------------------------------------------------------------------ Defaults
    def _set_defaults(self) -> None:
        pref = [
            "/zed_rear/zed_node/imu/data",
            "/ouster_rear/imu",
            "/ouster_front/imu",
        ]
        available = list(self.samples)
        sel = [p for p in pref if p in available]
        for t in available:
            if len(sel) >= 3:
                break
            if t not in sel:
                sel.append(t)
        self.active_topics = sel or available[:3]

    # ------------------------------------------------------------------ Plots
    def _draw_plots(self, verify: bool = False) -> None:
        self.fig.clear()
        self.ax_topic.clear()
        for sel in self.span_selector.values():
            sel.disconnect_events()
        self.span_selector.clear()

        rows = len(self.active_topics) * (2 if verify else 1)
        gs = self.fig.add_gridspec(rows, 1,
                                   height_ratios=[3, .6] * len(self.active_topics) if verify else None)

        for i, topic in enumerate(self.active_topics):
            df = self.dfs[topic]
            row = i * (2 if verify else 1)
            ax = self.fig.add_subplot(gs[row])
            ax.set_title(f"{topic} – Linear Acc.")
            if self.act_show_raw.isChecked():
                ax.plot(df["time"], df["ax"], label="ax", color="tab:blue")
                ax.plot(df["time"], df["ay"], label="ay", color="tab:orange")
                ax.plot(df["time"], df["az"], label="az", color="tab:green")
            if self.act_show_corr.isChecked() and "ax_corr" in df.columns:
                ax.plot(df["time"], df["ax_corr"], label="ax_corr", color="tab:blue", alpha=0.8, ls="--")
                ax.plot(df["time"], df["ay_corr"], label="ay_corr", color="tab:orange", alpha=0.8, ls="--")
                ax.plot(df["time"], df["az_corr"], label="az_corr", color="tab:green", alpha=0.8, ls="--")
            if self.act_show_veh.isChecked() and {"ax_veh", "ay_veh", "az_veh"}.issubset(df.columns):
                ax.plot(df["time"], df["ax_veh"], label="ax_veh", color="tab:blue", alpha=0.6, ls=":")
                ax.plot(df["time"], df["ay_veh"], label="ay_veh", color="tab:orange", alpha=0.6, ls=":")
                ax.plot(df["time"], df["az_veh"], label="az_veh", color="tab:green", alpha=0.6, ls=":")
            if row == rows - 1:
                ax.set_xlabel("Zeit ab Start [s]")
            ax.set_ylabel("m/s²")
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), loc="upper right")
            self.ax_topic[ax] = topic

            # Span-Selector
            self.span_selector[topic] = SpanSelector(
                ax,
                onselect=lambda xmin, xmax, t=topic: self.current_span.__setitem__(t, (xmin, xmax)),
                direction="horizontal",
                interactive=True,
                useblit=True,
                props=dict(alpha=.25, facecolor="tab:red"),
            )

            # Verify-Track
            if verify:
                ax_tr = self.fig.add_subplot(gs[row + 1], sharex=ax)
                for lname, props in ANOMALY_TYPES.items():
                    m = df["label_name"] == lname
                    if m.any():
                        ax_tr.scatter(df.loc[m, "time"], np.zeros(m.sum()),
                                      s=10, marker="s", color=props["color"], label=lname)
                unk = df["label_id"] == UNKNOWN_ID
                if unk.any():
                    ax_tr.scatter(df.loc[unk, "time"], np.zeros(unk.sum()),
                                  s=10, marker="s", color=UNKNOWN_COLOR, label=UNKNOWN_NAME)
                ax_tr.set_ylim(-1, 1)
                ax_tr.set_yticks([])
                ax_tr.set_xlabel("Zeit ab Start [s]")
                ax_tr.set_title("Label-Track")
                ax_tr.legend(fontsize="x-small", ncol=3, loc="upper right")

        self.canvas.draw_idle()

    # ------------------------------------------------------------------ Maus
    def _mouse_press(self, ev) -> None:
        if ev.button != 3 or ev.inaxes not in self.ax_topic:
            return
        topic = self.ax_topic[ev.inaxes]
        if topic not in self.current_span:
            return
        xmin, xmax = self.current_span[topic]
        if xmax <= xmin:
            return
        self._assign_label(topic, xmin, xmax)

    def _assign_label(self, topic: str, xmin: float, xmax: float) -> None:
        df = self.dfs[topic]
        lname = self.cmb.currentData()
        lid = self.LABEL_IDS[lname]
        color = ANOMALY_TYPES[lname]["color"]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        df.loc[mask, ["label_id", "label_name"]] = [lid, lname]

        ax = next(a for a, t in self.ax_topic.items() if t == topic)
        ax.axvspan(xmin, xmax, alpha=.2, color=color, label=lname)
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", ncol=2)
        self.canvas.draw_idle()

    def _check_export_status(self):
        rows = []
        bagstem = pathlib.Path(self.bag_path).stem
        for t, df in self.dfs.items():
            has_lbl = (df["label_id"] != 99).any()

            # Rotation: zuerst schauen, ob der Export schon lief und JSON vorliegt
            meta_file = (
                pathlib.Path(self.last_export_dir)
                / f"{t.strip('/').replace('/', '__')}_{bagstem}__imu_v1.json"
            ) if hasattr(self, "last_export_dir") else None

            if meta_file and meta_file.exists():
                rot_ok = json.loads(meta_file.read_text()).get("rotation_available", False)
            else:
                rot_ok = {"ax_veh", "ay_veh", "az_veh"}.issubset(df.columns) or \
                    (has_lbl and any(s.msg.orientation.w not in (0, 1) for s in self.samples[t]))

            rows.append((t, "✔" if has_lbl else "—", "✔" if rot_ok else "—"))
        html = "<table><tr><th>Topic</th><th>Labeled?</th><th>Rotation</th></tr>"
        for r in rows:
            html += f"<tr><td>{r[0]}</td><td align=center>{r[1]}</td>" \
                    f"<td align=center>{r[2]}</td></tr>"
        html += "</table>"
        QMessageBox.information(self, "Export Readiness", html)

    # ------------------------------------------------------------------ Settings-Dialog
    def _configure_topics(self) -> None:
        dlg = TopicDialog(list(self.samples), self.active_topics, self)
        if dlg.exec() != QDialog.Accepted:
            return
        sel = dlg.selected_topics()
        if not sel:
            QMessageBox.information(self, "Hinweis", "Mindestens ein Topic aktivieren.")
            return
        self.active_topics = sel
        self._draw_plots()



# ===========================================================================
def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
