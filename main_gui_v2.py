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
        QPushButton, QGroupBox, QRadioButton, QDoubleSpinBox, QTabWidget,
        QActionGroup
    )
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView
    except Exception:
        QWebEngineView = None
    from PyQt5.QtCore import Qt
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QFileDialog, QMessageBox,
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QAction, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
        QPushButton, QGroupBox, QRadioButton, QDoubleSpinBox, QTabWidget,
        QActionGroup
    )
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception:
        QWebEngineView = None
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
        _get_qt_widget,
    )
    from iso_weighting import calc_awv
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
    """Dialog to configure peak detection parameters."""

    def __init__(self, thr: float, dist: float, use_max: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Detection Settings")
        self.resize(300, 180)

        v = QVBoxLayout(self)

        form = QVBoxLayout()

        lbl_thr = QLabel("Peak threshold")
        self.sb_thr = QDoubleSpinBox()
        self.sb_thr.setRange(0.0, 100.0)
        self.sb_thr.setDecimals(2)
        self.sb_thr.setValue(thr)
        form.addWidget(lbl_thr)
        form.addWidget(self.sb_thr)

        lbl_dist = QLabel("Min. time between peaks [s]")
        self.sb_dist = QDoubleSpinBox()
        self.sb_dist.setRange(0.0, 10.0)
        self.sb_dist.setDecimals(2)
        self.sb_dist.setValue(dist)
        form.addWidget(lbl_dist)
        form.addWidget(self.sb_dist)

        self.chk_max = QCheckBox("Only mark maximum peak")
        self.chk_max.setChecked(use_max)
        form.addWidget(self.chk_max)

        v.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def result(self) -> tuple[float, float, bool]:
        return self.sb_thr.value(), self.sb_dist.value(), self.chk_max.isChecked()


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
        self.last_selected_topic: str | None = None
        self.label_patches: Dict[str, List[Tuple[float, float, object]]] = {}

        self.mount_overrides: dict[str, np.ndarray] = DEFAULT_OVERRIDES.copy()
        self.rot_mode: RotMode = RotMode.AUTO_ONLY
        self.iso_comfort: bool = True  # ISO weighting mode
        self.iso_metrics: dict[str, dict] = {}
        self.peak_threshold: float = 3.19
        self.peak_distance: float = 0.0
        self.use_max_peak: bool = False

        self._build_menu()
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        self.tabs = QTabWidget()
        vbox.addWidget(self.tabs)

        w_plot = QWidget()
        v_plot = QVBoxLayout(w_plot)

        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        v_plot.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._mouse_press)

        hl = QHBoxLayout()
        v_plot.addLayout(hl)
        hl.addWidget(QLabel("Segment Label:"))
        self.cmb = QComboBox()
        for n in ANOMALY_TYPES:
            self.cmb.addItem(n, userData=n)
        hl.addWidget(self.cmb)
        self.btn_add = QPushButton("Add")
        self.btn_edit = QPushButton("Edit")
        self.btn_del = QPushButton("Delete")
        self.btn_add.setMaximumWidth(60)
        self.btn_edit.setMaximumWidth(60)
        self.btn_del.setMaximumWidth(60)
        self.btn_add.clicked.connect(self._button_add_label)
        self.btn_edit.clicked.connect(self._button_edit_label)
        self.btn_del.clicked.connect(self._button_delete_label)
        hl.addWidget(self.btn_add)
        hl.addWidget(self.btn_edit)
        hl.addWidget(self.btn_del)
        hl.addStretch()

        self.tabs.addTab(w_plot, "Plots")

        w_map = QWidget()
        v_map = QVBoxLayout(w_map)
        if QWebEngineView is None:
            self.fig_map = Figure(constrained_layout=True)
            self.canvas_map = FigureCanvas(self.fig_map)
            v_map.addWidget(self.canvas_map)
        else:
            self.web_map = QWebEngineView()
            v_map.addWidget(self.web_map)
        self.tabs.addTab(w_map, "Map")

        self.tabs.currentChanged.connect(self._tab_changed)

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

        act_save_cfg = QAction("Save settings …", self)
        act_save_cfg.triggered.connect(self._save_config)
        m_file.addAction(act_save_cfg)

        act_load_cfg = QAction("Load settings …", self)
        act_load_cfg.triggered.connect(self._load_config)
        m_file.addAction(act_load_cfg)

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

        act_peaks = QAction("Peak detection …", self)
        act_peaks.triggered.connect(self._open_peak_dialog)
        m_imu.addAction(act_peaks)

        from PyQt5.QtWidgets import QActionGroup
        ag = QActionGroup(self)
        self.act_comfort = QAction("Comfort mode weighting", self, checkable=True)
        self.act_health = QAction("Health mode weighting", self, checkable=True)
        ag.addAction(self.act_comfort)
        ag.addAction(self.act_health)
        self.act_comfort.setChecked(self.iso_comfort)
        self.act_health.setChecked(not self.iso_comfort)
        self.act_comfort.triggered.connect(lambda: self._set_weighting(True))
        self.act_health.triggered.connect(lambda: self._set_weighting(False))
        m_imu.addAction(self.act_comfort)
        m_imu.addAction(self.act_health)

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

        self.act_show_iso = QAction("Show ISO weighted", self, checkable=True)
        self.act_show_iso.setChecked(False)
        self.act_show_iso.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_iso)

        self.act_show_peaks = QAction("Show ISO peaks", self, checkable=True)
        self.act_show_peaks.setChecked(True)
        self.act_show_peaks.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_peaks)

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
            gps_topic = next((t.name for t in topics_info if t.type == "sensor_msgs/msg/NavSatFix"), None)
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
        self._draw_map()

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
        self.iso_metrics.clear()
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

            # --- ISO 2631 weighting -----------------------------------------
            fs = 1.0 / np.median(np.diff(df["time"])) if len(df) > 1 else 0
            res = calc_awv(
                acc_corr[:, 0], acc_corr[:, 1], acc_corr[:, 2], fs,
                comfort=self.iso_comfort,
                peak_height=self.peak_threshold,
                peak_dist=self.peak_distance,
                max_peak=self.use_max_peak,
            )
            df[["awx", "awy", "awz"]] = np.column_stack(
                (res["awx"], res["awy"], res["awz"]))
            df[["rms_x", "rms_y", "rms_z"]] = np.column_stack(
                (res["rms_x"], res["rms_y"], res["rms_z"]))
            df["awv"] = res["awv"]
            self.iso_metrics[topic] = {
                "awv_total": res["awv_total"],
                "A8": res["A8"],
                "crest_factor": res["crest_factor"],
                "peaks": res["peaks"].tolist(),
            }

    def _resolve_rotation(self, topic: str, df: pd.DataFrame) -> np.ndarray | None:
        auto = auto_vehicle_frame(df, self._gps_df)
        ov = self.mount_overrides.get(topic)

        if self.rot_mode is RotMode.OVERRIDE_FIRST:
            if ov is not None:
                rot = ov
            else:
                rot = auto
        if self.rot_mode is RotMode.AUTO_FIRST:
            if auto is not None:
                rot = auto
            else:
                rot = ov
        else:
            rot = auto
        return np.asarray(rot) if rot is not None else None

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
        self.label_patches.clear()

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
            if self.act_show_iso.isChecked() and {"awx", "awy", "awz", "awv"}.issubset(df.columns):
                if self.act_show_x.isChecked():
                    ax.plot(df["time"], df["awx"], label="awx", color="tab:blue")
                if self.act_show_y.isChecked():
                    ax.plot(df["time"], df["awy"], label="awy", color="tab:orange")
                if self.act_show_z.isChecked():
                    ax.plot(df["time"], df["awz"], label="awz", color="tab:green")
                ax.plot(df["time"], df["awv"], label="awv", color="tab:red", lw=1.5)
                if self.act_show_peaks.isChecked():
                    peaks = self.iso_metrics.get(topic, {}).get("peaks", [])
                    if len(peaks):
                        ax.plot(df.loc[peaks, "time"], df.loc[peaks, "awv"], "k*", markersize=8, label="peaks")
            if row == rows - 1:
                ax.set_xlabel("Zeit ab Start [s]")
            ax.set_ylabel("m/s²")
            self._restore_labels(ax, topic)
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), loc="upper right")
            self.ax_topic[ax] = topic

            # Span-Selector
            self.span_selector[topic] = SpanSelector(
                ax,
                onselect=lambda xmin, xmax, t=topic: self._span_selected(t, xmin, xmax),
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
        if getattr(self, "tabs", None) and self.tabs.currentIndex() == 1:
            self._draw_map()

    def _span_selected(self, topic: str, xmin: float, xmax: float) -> None:
        self.current_span[topic] = (xmin, xmax)
        self.last_selected_topic = topic

    def _restore_labels(self, ax, topic: str) -> None:
        df = self.dfs[topic]
        self.label_patches[topic] = []
        if df.empty:
            return
        lbl = df["label_name"].to_numpy()
        times = df["time"].to_numpy()
        start = None
        last = UNKNOWN_NAME
        for t, name in zip(times, lbl):
            if name != last:
                if last != UNKNOWN_NAME and start is not None:
                    color = ANOMALY_TYPES[last]["color"]
                    patch = ax.axvspan(start, prev_t, alpha=.2, color=color, label=last)
                    self.label_patches[topic].append((start, prev_t, patch))
                start = t
                last = name
            prev_t = t
        if last != UNKNOWN_NAME and start is not None:
            color = ANOMALY_TYPES[last]["color"]
            patch = ax.axvspan(start, prev_t, alpha=.2, color=color, label=last)
            self.label_patches[topic].append((start, prev_t, patch))

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

    def _tab_changed(self, idx: int) -> None:
        if idx == 1:
            self._draw_map()

    def _draw_map(self) -> None:
        if QWebEngineView is None:
            self.fig_map.clear()
            if self._gps_df is None or not self.active_topics:
                self.canvas_map.draw_idle()
                return
            topic = self.active_topics[0]
            df = self.dfs[topic]
            merged = pd.merge_asof(
                df.sort_values("time_abs"),
                self._gps_df[["time", "lat", "lon"]].rename(columns={"time": "time_abs"}),
                on="time_abs",
                direction="nearest",
            )
            ax = self.fig_map.add_subplot(111)
            c = merged.get("awv", pd.Series(np.zeros(len(merged))))
            sc = ax.scatter(merged["lon"], merged["lat"], c=c, cmap="plasma", s=5)
            self.fig_map.colorbar(sc, ax=ax, label="awv")
            peaks = self.iso_metrics.get(topic, {}).get("peaks", [])
            if len(peaks):
                ax.scatter(merged.loc[peaks, "lon"], merged.loc[peaks, "lat"],
                           facecolors="none", edgecolors="black", s=40)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("GPS Track")
            self.canvas_map.draw_idle()
        else:
            if self._gps_df is None or not self.active_topics:
                self.web_map.setHtml("<p>No GPS data.</p>")
                return
            topic = self.active_topics[0]
            df = self.dfs[topic]
            merged = pd.merge_asof(
                df.sort_values("time_abs"),
                self._gps_df[["time", "lat", "lon"]].rename(columns={"time": "time_abs"}),
                on="time_abs",
                direction="nearest",
            )
            import folium
            try:
                import branca.colormap as cm
            except Exception:
                cm = None
            center = [merged["lat"].mean(), merged["lon"].mean()]
            fmap = folium.Map(location=center, zoom_start=15)
            cvals = merged.get("awv", pd.Series(np.zeros(len(merged))))
            if cm and hasattr(cm.linear, "Plasma_09"):
                colormap = cm.linear.Plasma_09.scale(float(cvals.min()), float(cvals.max()))
            elif cm:
                colormap = cm.LinearColormap(["blue", "red"], vmin=float(cvals.min()), vmax=float(cvals.max()))
            else:
                colormap = None
            for lat, lon, val in zip(merged["lat"], merged["lon"], cvals):
                color = colormap(float(val)) if colormap else None
                folium.CircleMarker(
                    location=[lat, lon], radius=3,
                    color=color, fill=True, fill_opacity=0.9,
                    fill_color=color
                ).add_to(fmap)
            if colormap:
                colormap.add_to(fmap)
            peaks = self.iso_metrics.get(topic, {}).get("peaks", [])
            for idx in peaks:
                if 0 <= idx < len(merged):
                    folium.CircleMarker(
                        location=[merged.loc[idx, "lat"], merged.loc[idx, "lon"]],
                        radius=6, color="black", fill=False
                    ).add_to(fmap)
            self.web_map.setHtml(fmap._repr_html_())

    def _assign_label(self, topic: str, xmin: float, xmax: float) -> None:
        df = self.dfs[topic]
        lname = self.cmb.currentData()
        lid = self.LABEL_IDS[lname]
        color = ANOMALY_TYPES[lname]["color"]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        df.loc[mask, ["label_id", "label_name"]] = [lid, lname]

        ax = next(a for a, t in self.ax_topic.items() if t == topic)
        patch = ax.axvspan(xmin, xmax, alpha=.2, color=color, label=lname)
        self.label_patches.setdefault(topic, []).append((xmin, xmax, patch))
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", ncol=2)
        self.canvas.draw_idle()

    def _delete_label_range(self, topic: str, xmin: float, xmax: float) -> None:
        df = self.dfs[topic]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        df.loc[mask, ["label_id", "label_name"]] = [UNKNOWN_ID, UNKNOWN_NAME]

        if topic in self.label_patches:
            new_list = []
            for start, end, patch in self.label_patches[topic]:
                if end <= xmin or start >= xmax:
                    new_list.append((start, end, patch))
                else:
                    patch.remove()
            self.label_patches[topic] = new_list
        self.canvas.draw_idle()

    def _button_add_label(self) -> None:
        t = self.last_selected_topic
        if not t or t not in self.current_span:
            return
        xmin, xmax = self.current_span[t]
        if xmax <= xmin:
            return
        self._assign_label(t, xmin, xmax)

    def _button_edit_label(self) -> None:
        t = self.last_selected_topic
        if not t or t not in self.current_span:
            return
        xmin, xmax = self.current_span[t]
        if xmax <= xmin:
            return
        self._delete_label_range(t, xmin, xmax)
        self._assign_label(t, xmin, xmax)

    def _button_delete_label(self) -> None:
        t = self.last_selected_topic
        if not t or t not in self.current_span:
            return
        xmin, xmax = self.current_span[t]
        self._delete_label_range(t, xmin, xmax)

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
        self._draw_plots()

    def _open_peak_dialog(self) -> None:
        dlg = PeakDialog(self.peak_threshold, self.peak_distance, self.use_max_peak, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self.peak_threshold, self.peak_distance, self.use_max_peak = dlg.result()
        self._preprocess_all()
        self._draw_plots()

    def _set_weighting(self, comfort: bool) -> None:
        self.iso_comfort = comfort
        self.act_comfort.setChecked(comfort)
        self.act_health.setChecked(not comfort)
        self._preprocess_all()
        self._draw_plots()

    def _save_config(self) -> None:
        QFileDialog = _get_qt_widget(self, "QFileDialog")
        if QFileDialog is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Settings", str(pathlib.Path.cwd()), "JSON Files (*.json)")
        if not path:
            return
        cfg = {
            "mount_overrides": {k: v.tolist() for k, v in self.mount_overrides.items()},
            "rot_mode": self.rot_mode.name,
            "peak_threshold": self.peak_threshold,
            "peak_distance": self.peak_distance,
            "use_max_peak": self.use_max_peak,
            "iso_comfort": self.iso_comfort,
        }
        pathlib.Path(path).write_text(json.dumps(cfg, indent=2))

    def _load_config(self) -> None:
        QFileDialog = _get_qt_widget(self, "QFileDialog")
        if QFileDialog is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Settings", str(pathlib.Path.cwd()), "JSON Files (*.json)")
        if not path:
            return
        data = json.loads(pathlib.Path(path).read_text())
        self.mount_overrides = {k: np.array(v) for k, v in data.get("mount_overrides", {}).items()}
        self.rot_mode = RotMode[data.get("rot_mode", self.rot_mode.name)]
        self.peak_threshold = float(data.get("peak_threshold", self.peak_threshold))
        self.peak_distance = float(data.get("peak_distance", self.peak_distance))
        self.use_max_peak = bool(data.get("use_max_peak", self.use_max_peak))
        self.iso_comfort = bool(data.get("iso_comfort", self.iso_comfort))
        self.act_comfort.setChecked(self.iso_comfort)
        self.act_health.setChecked(not self.iso_comfort)
        self._preprocess_all()
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
        self._draw_plots()



# ===========================================================================
def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
