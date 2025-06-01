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
from typing import Dict, List, Tuple, cast
from enum import Enum, auto

import numpy as np
import pandas as pd
import json
import os
from scipy.signal import find_peaks

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
        QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
        QPushButton, QGroupBox, QRadioButton, QDoubleSpinBox,
    )
    from PyQt5.QtCore import Qt
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QFileDialog, QMessageBox,
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QAction, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
        QPushButton, QGroupBox, QRadioButton, QDoubleSpinBox,
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
# Rotation Modes & Default Overrides
# ===========================================================================
class RotMode(Enum):
    OVERRIDE_FIRST = auto()
    AUTO_FIRST = auto()
    AUTO_ONLY = auto()

# Default-Overrides (kann der User gleich im Dialog ändern)
DEFAULT_OVERRIDES: dict[str, np.ndarray] = {
    "/zed_right/zed_node/imu/data": np.array([[0, 1, 0],
                                              [-1, 0, 0],
                                              [0, 0, 1]], float),
    "/zed_left/zed_node/imu/data": np.array([[0, -1, 0],
                                             [1, 0, 0],
                                             [0, 0, 1]], float),
}

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
# Mount-Dialog
# ===========================================================================
class MountDialog(QDialog):
    """Dialog: Mounting-Override verwalten + Vorschau."""

    def __init__(self, topics: list[str],
                 overrides: dict[str, np.ndarray],
                 mode: RotMode, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mounting Overrides")
        self.resize(680, 520)

        v = QVBoxLayout(self)

        # Strategiewahl
        grp_mode = QGroupBox("Anwendungs-Strategie")
        rb1 = QRadioButton("Override ⟶ Auto (Override-first)")
        rb2 = QRadioButton("Auto ⟶ Override (Auto-first)")
        rb3 = QRadioButton("Nur Auto (Override deaktiviert)")
        if mode is RotMode.OVERRIDE_FIRST:
            rb1.setChecked(True)
        elif mode is RotMode.AUTO_FIRST:
            rb2.setChecked(True)
        else:
            rb3.setChecked(True)
        hlm = QVBoxLayout(grp_mode)
        hlm.addWidget(rb1)
        hlm.addWidget(rb2)
        hlm.addWidget(rb3)
        v.addWidget(grp_mode)

        # Topic-Tabelle
        self.tbl = QTableWidget(len(topics), 5)
        self.tbl.setHorizontalHeaderLabels([
            "Topic", "Override aktiv", "Edit", "Ist-Achsen", "Neu-Achsen"
        ])
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for row, t in enumerate(topics):
            item = QTableWidgetItem(t)
            item.setFlags(Qt.ItemIsEnabled)
            self.tbl.setItem(row, 0, item)

            chk = QCheckBox()
            chk.setChecked(t in overrides)
            self.tbl.setCellWidget(row, 1, chk)

            btn = QPushButton("⚙")
            btn.clicked.connect(lambda _=None, topic=t: self._edit_matrix(topic))
            self.tbl.setCellWidget(row, 2, btn)

            self.tbl.setCellWidget(row, 3, self._axes_widget(np.eye(3)))
            R0 = overrides.get(t, np.eye(3))
            self.tbl.setCellWidget(row, 4, self._axes_widget(R0))
        v.addWidget(self.tbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        self.rb = (rb1, rb2, rb3)
        self.topics = topics
        self.overrides = overrides

    @staticmethod
    def _axes_widget(R: np.ndarray):
        fig = Figure(figsize=(1.6, 1.6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        origin = np.zeros((3, 1))
        ax.quiver(*origin, *R[:, 0], color="r")
        ax.quiver(*origin, *R[:, 1], color="g")
        ax.quiver(*origin, *R[:, 2], color="b")
        return canvas

    def _edit_matrix(self, topic: str):
        R0 = self.overrides.get(topic, np.eye(3))
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Override für {topic}")
        l = QVBoxLayout(dlg)
        tbl = QTableWidget(3, 3)
        for i in range(3):
            for j in range(3):
                it = QTableWidgetItem(f"{R0[i, j]:.3f}")
                tbl.setItem(i, j, it)
        l.addWidget(tbl)
        l.addWidget(QLabel("Einheit: Richtungskosinus"))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        l.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        R_new = np.array([[float(tbl.item(i, j).text()) for j in range(3)] for i in range(3)])
        self.overrides[topic] = R_new
        self.tbl.setCellWidget(self._row_of(topic), 4, self._axes_widget(R_new))

    def _row_of(self, topic):
        return next(i for i, t in enumerate(self.topics) if t == topic)

    def result(self):
        if self.rb[0].isChecked():
            mode = RotMode.OVERRIDE_FIRST
        elif self.rb[1].isChecked():
            mode = RotMode.AUTO_FIRST
        else:
            mode = RotMode.AUTO_ONLY

        ov: dict[str, np.ndarray] = {}
        for row, t in enumerate(self.topics):
            if cast(QCheckBox, self.tbl.cellWidget(row, 1)).isChecked():
                ov[t] = self.overrides.get(t, np.eye(3))
        return mode, ov


# ===========================================================================
# Peak Settings Dialog
# ===========================================================================
class PeakDialog(QDialog):
    def __init__(self, threshold: float, min_dt: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Detection Settings")

        v = QVBoxLayout(self)

        hl1 = QHBoxLayout()
        hl1.addWidget(QLabel("Threshold [m/s²]"))
        self.sp_thresh = QDoubleSpinBox()
        self.sp_thresh.setRange(0.0, 100.0)
        self.sp_thresh.setDecimals(2)
        self.sp_thresh.setValue(threshold)
        self.sp_thresh.setSingleStep(0.1)
        hl1.addWidget(self.sp_thresh)
        v.addLayout(hl1)

        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Min. distance [s]"))
        self.sp_dist = QDoubleSpinBox()
        self.sp_dist.setRange(0.0, 10.0)
        self.sp_dist.setDecimals(2)
        self.sp_dist.setValue(min_dt)
        self.sp_dist.setSingleStep(0.1)
        hl2.addWidget(self.sp_dist)
        v.addLayout(hl2)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def result(self) -> tuple[float, float]:
        return self.sp_thresh.value(), self.sp_dist.value()


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

        self.peak_threshold: float = 5.0
        self.peak_min_dt: float = 0.5
        self.peaks: dict[str, pd.DataFrame] = {}

        self.mount_overrides: dict[str, np.ndarray] = DEFAULT_OVERRIDES.copy()
        self.rot_mode: RotMode = RotMode.AUTO_ONLY

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

        act_mount = QAction("Mounting Overrides …", self)
        act_mount.triggered.connect(self._open_mount_dialog)
        m_imu.addAction(act_mount)

        self.act_peaks = QAction("Peak Detection …", self)
        self.act_peaks.setEnabled(False)
        self.act_peaks.triggered.connect(self._configure_peaks)
        m_imu.addAction(self.act_peaks)

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

        # Axis visibility
        self.act_show_x = QAction("Show X axis", self, checkable=True)
        self.act_show_x.setChecked(True)
        self.act_show_x.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_x)

        self.act_show_y = QAction("Show Y axis", self, checkable=True)
        self.act_show_y.setChecked(True)
        self.act_show_y.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_y)

        self.act_show_z = QAction("Show Z axis", self, checkable=True)
        self.act_show_z.setChecked(True)
        self.act_show_z.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_z)

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
        self._update_all_peaks()
        self._set_defaults()
        self._draw_plots()

        self.act_topics.setEnabled(True)
        self.act_verify.setEnabled(True)
        self.act_export.setEnabled(True)
        self.act_check.setEnabled(True)
        self.act_peaks.setEnabled(True)

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
            rot = self._resolve_rotation(topic, df)
            if rot is not None:
                R = np.asarray(rot)
                veh = acc_corr @ R.T
                df[["ax_veh", "ay_veh", "az_veh"]] = veh

    def _resolve_rotation(self, topic: str, df: pd.DataFrame) -> np.ndarray | None:
        auto = auto_vehicle_frame(df, self._gps_df)
        ov = self.mount_overrides.get(topic)

        if self.rot_mode is RotMode.OVERRIDE_FIRST:
            if ov is not None:
                return ov
            return auto
        if self.rot_mode is RotMode.AUTO_FIRST:
            if auto is not None:
                return auto
            return ov
        return auto

    # ------------------------------------------------------------------ Peaks
    def _detect_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["ax_corr", "ay_corr", "az_corr"] if {"ax_corr", "ay_corr", "az_corr"}.issubset(df.columns) else ["ax", "ay", "az"]
        acc = df[cols].to_numpy()
        mag = np.linalg.norm(acc, axis=1)
        if len(mag) < 2:
            return pd.DataFrame(columns=["time", "value", "time_abs"])
        dt = np.median(np.diff(df["time"]))
        min_samples = max(1, int(self.peak_min_dt / dt)) if dt > 0 else 1
        idx, _ = find_peaks(mag, height=self.peak_threshold, distance=min_samples)
        return pd.DataFrame({
            "time": df["time"].iloc[idx].to_numpy(),
            "value": mag[idx],
            "time_abs": df["time_abs"].iloc[idx].to_numpy(),
        })

    def _update_all_peaks(self) -> None:
        self.peaks = {t: self._detect_peaks(df) for t, df in self.dfs.items()}

    # ------------------------------------------------------------------ Defaults
    def _set_defaults(self) -> None:
        pref = [
            "/zed_rear/zed_node/imu/data",
            "/zed_left/zed_node/imu/data",
            "/zed_right/zed_node/imu/data",
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
                if self.act_show_x.isChecked():
                    ax.plot(df["time"], df["ax"], label="ax", color="tab:blue")
                if self.act_show_y.isChecked():
                    ax.plot(df["time"], df["ay"], label="ay", color="tab:orange")
                if self.act_show_z.isChecked():
                    ax.plot(df["time"], df["az"], label="az", color="tab:green")
            if self.act_show_corr.isChecked() and "ax_corr" in df.columns:
                if self.act_show_x.isChecked():
                    ax.plot(df["time"], df["ax_corr"], label="ax_corr", color="tab:blue", alpha=0.8, ls="--")
                if self.act_show_y.isChecked():
                    ax.plot(df["time"], df["ay_corr"], label="ay_corr", color="tab:orange", alpha=0.8, ls="--")
                if self.act_show_z.isChecked():
                    ax.plot(df["time"], df["az_corr"], label="az_corr", color="tab:green", alpha=0.8, ls="--")
            if self.act_show_veh.isChecked() and {"ax_veh", "ay_veh", "az_veh"}.issubset(df.columns):
                if self.act_show_x.isChecked():
                    ax.plot(df["time"], df["ax_veh"], label="ax_veh", color="tab:blue", alpha=0.6, ls=":")
                if self.act_show_y.isChecked():
                    ax.plot(df["time"], df["ay_veh"], label="ay_veh", color="tab:orange", alpha=0.6, ls=":")
                if self.act_show_z.isChecked():
                    ax.plot(df["time"], df["az_veh"], label="az_veh", color="tab:green", alpha=0.6, ls=":")
            peaks_df = self.peaks.get(topic)
            if peaks_df is not None and not peaks_df.empty:
                ax.plot(peaks_df["time"], peaks_df["value"], "k*", label="peaks")
                for t_val, y_val, ta in zip(peaks_df["time"], peaks_df["value"], peaks_df["time_abs"]):
                    ax.text(t_val, y_val, f"{ta:.1f}", fontsize="x-small", rotation=90, va="bottom")
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

    def _open_mount_dialog(self) -> None:
        dlg = MountDialog(list(self.samples), self.mount_overrides.copy(), self.rot_mode, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.rot_mode, self.mount_overrides = dlg.result()
        self._preprocess_all()
        self._update_all_peaks()
        self._draw_plots()

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
        self._update_all_peaks()
        self._draw_plots()

    def _configure_peaks(self) -> None:
        dlg = PeakDialog(self.peak_threshold, self.peak_min_dt, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.peak_threshold, self.peak_min_dt = dlg.result()
        self._update_all_peaks()
        self._draw_plots()



# ===========================================================================
def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
