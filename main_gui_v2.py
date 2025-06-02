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
from concurrent.futures import ProcessPoolExecutor
from enum import Enum, auto
import logging

import numpy as np
import pandas as pd
import json
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('debug.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

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
        QActionGroup, QUndoStack, QUndoCommand
    )
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView
    except Exception:
        QWebEngineView = None
    from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QKeySequence, QPainter, QImage
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QFileDialog, QMessageBox,
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
        QAction, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
        QPushButton, QGroupBox, QRadioButton, QDoubleSpinBox, QTabWidget,
        QActionGroup, QUndoStack, QUndoCommand
    )
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception:
        QWebEngineView = None
    from PySide6.QtCore import Qt, QSettings, QThread, Signal as pyqtSignal, QTimer
    from PySide6.QtGui import QKeySequence, QPainter, QImage

# ---------------------------------------------------------------------------#
# ROS-Import                                                                 #
# ---------------------------------------------------------------------------#
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import (
        Imu, NavSatFix, Image, CompressedImage, PointCloud2
    )
    from cv_bridge import CvBridge
    from sensor_msgs_py import point_cloud2 as pc2
    import cv2
    from imu_csv_export_v2 import (
        export_csv_smart_v2,
        remove_gravity_lowpass,
        auto_vehicle_frame,
        gravity_from_quat,
        _get_qt_widget,
    )
    from iso_weighting import calc_awv
    from progress_ui import ProgressWindow
    from videopc_widget import VideoPointCloudTab
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
    # ZED rear camera: flipped by 180° around X and Y (equals 180° around Z)
    "/zed_rear/zed_node/imu/data": np.array([[-1, 0, 0],
                                             [0, -1, 0],
                                             [0, 0, 1]], float),

    # Ouster rear lidar IMU: same orientation as the rear ZED
    "/ouster_rear/imu": np.array([[-1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 1]], float),

    # ZED left camera: rotate 90° to the right (clockwise around Z)
    "/zed_left/zed_node/imu/data": np.array([[0, 1, 0],
                                             [-1, 0, 0],
                                             [0, 0, 1]], float),

    # ZED right camera: rotate 90° to the left (counter-clockwise around Z)
    "/zed_right/zed_node/imu/data": np.array([[0, -1, 0],
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


def _preprocess_single(args):
    """Helper for parallel preprocessing of one topic."""
    (
        topic,
        df,
        acc_arr,
        ori_arr,
        gps_df,
        rot_mode,
        overrides,
        iso_comfort,
        peak_thr,
        peak_dist,
        use_max,
    ) = args

    ori = ori_arr
    norm_ok = np.abs(np.linalg.norm(ori, axis=1) - 1.0) < 0.05
    var_ok = np.ptp(ori, axis=0).max() > 1e-3
    has_quat = bool(norm_ok.any() and var_ok)

    if has_quat:
        g_vec = gravity_from_quat(pd.DataFrame(ori, columns=["ox", "oy", "oz", "ow"]))
        acc_corr = acc_arr - g_vec
        g_est = g_vec
    else:
        acc_corr, g_est, _ = remove_gravity_lowpass(df)

    df[["accel_corr_x", "accel_corr_y", "accel_corr_z"]] = acc_corr
    df[["grav_x", "grav_y", "grav_z"]] = g_est

    auto = auto_vehicle_frame(df, gps_df)
    ov = overrides.get(topic)
    if rot_mode == RotMode.OVERRIDE_FIRST:
        rot = ov if ov is not None else auto
    elif rot_mode == RotMode.AUTO_FIRST:
        rot = auto if auto is not None else ov
    else:
        rot = auto
    if rot is not None:
        R = np.asarray(rot)
        veh = acc_corr @ R.T
        df[["accel_veh_x", "accel_veh_y", "accel_veh_z"]] = veh

    fs = 1.0 / np.median(np.diff(df["time"])) if len(df) > 1 else 0
    res = calc_awv(
        acc_corr[:, 0], acc_corr[:, 1], acc_corr[:, 2], fs,
        comfort=iso_comfort,
        peak_height=peak_thr,
        peak_dist=peak_dist,
        max_peak=use_max,
    )
    df[["awx", "awy", "awz"]] = np.column_stack((res["awx"], res["awy"], res["awz"]))
    df[["rms_x", "rms_y", "rms_z"]] = np.column_stack((res["rms_x"], res["rms_y"], res["rms_z"]))
    df["awv"] = res["awv"]
    metrics = {
        "awv_total": res["awv_total"],
        "A8": res["A8"],
        "crest_factor": res["crest_factor"],
        "peaks": res["peaks"].tolist(),
    }
    return topic, df, metrics


class BagReaderWorker(QThread):
    """Worker-Thread zum asynchronen Einlesen der Bag-Datei."""

    stepChanged = pyqtSignal(str)
    setMaximum  = pyqtSignal(int)
    progress    = pyqtSignal(int)
    finished    = pyqtSignal(object, object)

    def __init__(self, bag_path: pathlib.Path) -> None:
        super().__init__()
        self.bag_path = bag_path

    def run(self) -> None:
        try:
            log.debug("BagReaderWorker started")
            self.stepChanged.emit("Öffne Bag-Datei …")
            reader = SequentialReader()
            reader.open(
                StorageOptions(str(self.bag_path), "sqlite3"),
                ConverterOptions("cdr", "cdr"),
            )

            self.stepChanged.emit("Ermittle Topics …")
            topics_info = reader.get_all_topics_and_types()
            topic_types = {t.name: t.type for t in topics_info}
            imu_topics = [t for t, ty in topic_types.items() if ty == "sensor_msgs/msg/Imu"]
            gps_topic = next((t for t, ty in topic_types.items() if ty == "sensor_msgs/msg/NavSatFix"), None)
            image_topics = [t for t, ty in topic_types.items() if ty in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")]
            pc_topics = [t for t, ty in topic_types.items() if ty == "sensor_msgs/msg/PointCloud2"]

            total = reader.get_metadata().message_count
            self.setMaximum.emit(total)

            samples = {t: [] for t in imu_topics}
            gps: list[tuple] = []
            video_frames_by_topic: dict[str, list[bytes | memoryview]] = {t: [] for t in image_topics}
            video_times_by_topic: dict[str, list[float]] = {t: [] for t in image_topics}
            pc_frames_by_topic: dict[str, list[PointCloud2]] = {t: [] for t in pc_topics}
            pc_times_by_topic: dict[str, list[float]] = {t: [] for t in pc_topics}
            cnt = 0
            bridge = CvBridge()

            self.stepChanged.emit("Lese Daten …")
            while reader.has_next():
                topic, data, ts = reader.read_next()
                if topic in samples:
                    samples[topic].append(ImuSample(ts, deserialize_message(data, Imu)))
                elif gps_topic and topic == gps_topic:
                    msg = deserialize_message(data, NavSatFix)
                    gps.append((ts / 1e9, msg.latitude, msg.longitude, msg.altitude))
                elif topic in video_frames_by_topic:
                    mtype = topic_types[topic]
                    if mtype == "sensor_msgs/msg/Image":
                        img_msg = deserialize_message(data, Image)
                        video_frames_by_topic[topic].append(memoryview(img_msg.data))
                    else:
                        img_msg = deserialize_message(data, CompressedImage)
                        video_frames_by_topic[topic].append(bytes(img_msg.data))
                    video_times_by_topic[topic].append(ts / 1e9)
                elif topic in pc_frames_by_topic:
                    pc_msg = deserialize_message(data, PointCloud2)
                    pc_frames_by_topic[topic].append(pc_msg)
                    pc_times_by_topic[topic].append(ts / 1e9)
                cnt += 1
                if cnt % 200 == 0:
                    self.progress.emit(cnt)

            self.progress.emit(total)
            available = [t for t, vals in samples.items() if vals]
            log.debug("BagReaderWorker finished reading")
            self.finished.emit(
                (
                    samples,
                    gps,
                    available,
                    gps_topic,
                    image_topics,
                    pc_topics,
                    video_frames_by_topic,
                    pc_frames_by_topic,
                    video_times_by_topic,
                    pc_times_by_topic,
                ),
                None,
            )
        except Exception as exc:
            log.exception("BagReaderWorker error")
            self.finished.emit(None, exc)


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
        if mode == RotMode.OVERRIDE_FIRST:
            rb1.setChecked(True)
        elif mode == RotMode.AUTO_FIRST:
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


class LabelManagerDialog(QDialog):
    """Advanced dialog to edit existing label segments."""

    def __init__(self, dfs: dict[str, pd.DataFrame], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Labels")
        self.resize(600, 400)

        self.dfs = dfs
        self.tbl = QTableWidget(self)
        self.tbl.setColumnCount(4)
        self.tbl.setHorizontalHeaderLabels(["Topic", "Label", "Start", "End"])

        rows = []
        for topic, df in dfs.items():
            mask = df["label_name"] != UNKNOWN_NAME
            if not mask.any():
                continue
            seg_id = (df["label_name"] != df["label_name"].shift()).cumsum()
            for _, grp in df[mask].groupby(seg_id):
                rows.append((topic, grp["label_name"].iat[0], grp["time"].iat[0], grp["time"].iat[-1]))

        self.tbl.setRowCount(len(rows))
        for i, (t, lbl, s, e) in enumerate(rows):
            self.tbl.setItem(i, 0, QTableWidgetItem(t))
            it_lbl = QTableWidgetItem(lbl)
            it_lbl.setFlags(it_lbl.flags() | Qt.ItemIsEditable)
            self.tbl.setItem(i, 1, it_lbl)
            self.tbl.setItem(i, 2, QTableWidgetItem(f"{s:.2f}"))
            self.tbl.setItem(i, 3, QTableWidgetItem(f"{e:.2f}"))

        layout = QVBoxLayout(self)
        layout.addWidget(self.tbl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def result(self) -> dict[str, list[tuple[str, float, float]]]:
        res: dict[str, list[tuple[str, float, float]]] = {}
        for r in range(self.tbl.rowCount()):
            topic = self.tbl.item(r, 0).text()
            label = self.tbl.item(r, 1).text()
            s = float(self.tbl.item(r, 2).text())
            e = float(self.tbl.item(r, 3).text())
            res.setdefault(topic, []).append((label, s, e))
        return res


# ===========================================================================
# Undo/Redo Commands
# ===========================================================================
class AddLabelCmd(QUndoCommand):
    """Undoable command: add label to a time range."""

    def __init__(self, win, topic: str, xmin: float, xmax: float, lname: str):
        super().__init__(f"Add {lname}")
        self.win = win
        self.topic = topic
        self.xmin = xmin
        self.xmax = xmax
        self.lname = lname
        df = win.dfs[topic]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        self.prev = df.loc[mask, ["label_id", "label_name"]].copy()

    def redo(self) -> None:  # type: ignore[override]
        self.win._assign_label(self.topic, self.xmin, self.xmax, self.lname)
        self.win._draw_plots(self.win.act_verify.isChecked())

    def undo(self) -> None:  # type: ignore[override]
        df = self.win.dfs[self.topic]
        mask = (df["time"] >= self.xmin) & (df["time"] <= self.xmax)
        df.loc[mask, ["label_id", "label_name"]] = self.prev.values
        ax = next(a for a, t in self.win.ax_topic.items() if t == self.topic)
        self.win._restore_labels(ax, self.topic)
        self.win._draw_plots(self.win.act_verify.isChecked())


class DeleteLabelCmd(QUndoCommand):
    """Undoable command: delete labels in a range."""

    def __init__(self, win, topic: str, xmin: float, xmax: float):
        super().__init__("Delete labels")
        self.win = win
        self.topic = topic
        self.xmin = xmin
        self.xmax = xmax
        df = win.dfs[topic]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        self.prev = df.loc[mask, ["label_id", "label_name"]].copy()

    def redo(self) -> None:  # type: ignore[override]
        self.win._delete_label_range(self.topic, self.xmin, self.xmax)
        self.win._draw_plots(self.win.act_verify.isChecked())

    def undo(self) -> None:  # type: ignore[override]
        df = self.win.dfs[self.topic]
        mask = (df["time"] >= self.xmin) & (df["time"] <= self.xmax)
        df.loc[mask, ["label_id", "label_name"]] = self.prev.values
        ax = next(a for a, t in self.win.ax_topic.items() if t == self.topic)
        self.win._restore_labels(ax, self.topic)
        self.win._draw_plots(self.win.act_verify.isChecked())


class EditLabelCmd(QUndoCommand):
    """Undoable command: change label in a range."""

    def __init__(self, win, topic: str, xmin: float, xmax: float, lname: str):
        super().__init__(f"Edit → {lname}")
        self.win = win
        self.topic = topic
        self.xmin = xmin
        self.xmax = xmax
        self.lname = lname
        df = win.dfs[topic]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        self.prev = df.loc[mask, ["label_id", "label_name"]].copy()

    def redo(self) -> None:  # type: ignore[override]
        self.win._delete_label_range(self.topic, self.xmin, self.xmax)
        self.win._assign_label(self.topic, self.xmin, self.xmax, self.lname)
        self.win._draw_plots(self.win.act_verify.isChecked())

    def undo(self) -> None:  # type: ignore[override]
        df = self.win.dfs[self.topic]
        mask = (df["time"] >= self.xmin) & (df["time"] <= self.xmax)
        df.loc[mask, ["label_id", "label_name"]] = self.prev.values
        ax = next(a for a, t in self.win.ax_topic.items() if t == self.topic)
        self.win._restore_labels(ax, self.topic)
        self.win._draw_plots(self.win.act_verify.isChecked())

# ===========================================================================
# Main-Window
# ===========================================================================
class MainWindow(QMainWindow):
    LABEL_IDS = {name: i + 1 for i, name in enumerate(ANOMALY_TYPES)}

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROS 2 IMU-Labeling Tool")
        self.resize(1500, 900)

        # Persistent settings
        self.settings = QSettings("FH-Zürich", "IMU-LabelTool")
        self.undo = QUndoStack(self)

        # Daten-Strukturen
        self.bag_path: pathlib.Path | None = None
        self.samples: dict[str, list[ImuSample]] = {}
        self.dfs: dict[str, pd.DataFrame] = {}
        self.t0: float | None = None
        self._gps_df: pd.DataFrame | None = None
        self.available_topics: list[str] = []

        # Runtime-State
        self.active_topics: list[str] = []
        self.ax_topic: dict[object, str] = {}
        self.span_selector: dict[str, SpanSelector] = {}
        self.current_span: dict[str, Tuple[float, float]] = {}
        self.last_selected_topic: str | None = None
        self.label_patches: Dict[str, List[Tuple[float, float, object]]] = {}
        # Track-Segmente zur Synchronisation von Verifikations-Ansicht
        self.label_patches_track: Dict[str, List[Tuple[float, float, str]]] = {}

        self.mount_overrides: dict[str, np.ndarray] = DEFAULT_OVERRIDES.copy()
        self.rot_mode: RotMode = RotMode.OVERRIDE_FIRST
        self.iso_comfort: bool = True  # ISO weighting mode
        self.iso_metrics: dict[str, dict] = {}
        self.peak_threshold: float = 3.19
        self.peak_distance: float = 1.5
        self.use_max_peak: bool = False

        # Video/PointCloud playback
        self.video_topic: str | None = None
        self.pc_topic: str | None = None
        self.video_frames_by_topic: dict[str, list] = {}
        self.video_times_by_topic: dict[str, list[float]] = {}
        self.pc_frames_by_topic: dict[str, list] = {}
        self.pc_times_by_topic: dict[str, list[float]] = {}

        self._build_menu()
        self._build_ui()

        # Restore persisted window state
        self._restore_settings()

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

        # ------------------------------------------------------ Videos + PC
        self.tab_vpc = VideoPointCloudTab()
        self.tabs.addTab(self.tab_vpc, "Videos + PC")
        self.tab_vpc.cmb_video.currentTextChanged.connect(self._change_video_topic)
        self.tab_vpc.cmb_pc.currentTextChanged.connect(self._change_pc_topic)

        self.tabs.currentChanged.connect(self._tab_changed)

    # ------------------------------------------------------------------ Settings
    def _restore_settings(self) -> None:
        geom = self.settings.value("geom", b"")
        if isinstance(geom, (bytes, bytearray)):
            self.restoreGeometry(geom)
        state = self.settings.value("state", b"")
        if isinstance(state, (bytes, bytearray)):
            self.restoreState(state)
        topics = self.settings.value("active_topics")
        if isinstance(topics, list):
            self.active_topics = [str(t) for t in topics]

    def closeEvent(self, e) -> None:
        self.settings.setValue("geom", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.settings.setValue("active_topics", self.active_topics)
        super().closeEvent(e)

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

        # Edit menu with Undo/Redo
        m_edit = mb.addMenu("&Edit")
        act_undo = self.undo.createUndoAction(self, "Undo")
        act_redo = self.undo.createRedoAction(self, "Redo")
        act_undo.setShortcut(QKeySequence.Undo)
        act_redo.setShortcut(QKeySequence.Redo)
        m_edit.addAction(act_undo)
        m_edit.addAction(act_redo)

        act_manage = QAction("Manage labels …", self)
        act_manage.triggered.connect(self._open_label_manager)
        m_edit.addAction(act_manage)

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
        self.act_show_raw.setChecked(False)
        self.act_show_raw.triggered.connect(lambda: self._draw_plots())
        m_view.addAction(self.act_show_raw)

        self.act_show_corr = QAction("Show g-corrected acceleration", self, checkable=True)
        self.act_show_corr.setChecked(False)
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
        self.act_show_iso.setChecked(True)
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
        self.available_topics.clear()

        steps = [
            "Öffne Bag-Datei",
            "Ermittle Topics",
            "Lese Daten",
            "Erzeuge DataFrames",
            "Vorverarbeitung (g-Korrektur, ISO-Weights …)",
            "Erstelle Plots",
            "Erstelle Karte",
        ]
        progress = ProgressWindow("Bag einlesen", steps, parent=self)

        self.worker = BagReaderWorker(self.bag_path)
        self.worker.stepChanged.connect(progress.advance)
        self.worker.setMaximum.connect(progress.set_bar_range)
        self.worker.progress.connect(progress.set_bar_value)
        self.worker.finished.connect(lambda res, err: self._reader_done(res, err, progress))
        self.worker.start()

    def _reader_done(self, result: tuple | None, err: Exception | None, progress: ProgressWindow) -> None:
        """Callback nach dem asynchronen Lese-Vorgang."""
        log.debug("Reader done callback invoked")
        if progress.wasCanceled():
            log.debug("Progress dialog was canceled")
            progress.accept()
            return
        if err:
            log.error("Error from worker: %s", err)
            QMessageBox.critical(self, "Lesefehler", str(err))
            progress.accept()
            return

        assert result is not None
        (samples, gps, available, gps_topic,
         image_topics, pc_topics,
         vid_frames, pc_frames,
         vid_times, pc_times) = result
        if not available:
            QMessageBox.information(self, "Keine IMU-Topics", "Es wurden keine IMU-Topics gefunden.")
            progress.accept()
            return

        # nur Topics mit Daten behalten
        self.samples = cast(dict[str, list[ImuSample]], {t: samples[t] for t in available})
        self.available_topics = available
        self._gps_df = pd.DataFrame(gps, columns=["time", "lat", "lon", "alt"]) if gps else None
        self.video_frames_by_topic = vid_frames
        self.video_times_by_topic = vid_times
        self.pc_frames_by_topic = pc_frames
        self.pc_times_by_topic = pc_times
        log.debug("Available topics: %s", self.available_topics)

        self.tab_vpc.cmb_video.clear()
        self.tab_vpc.cmb_video.addItems(image_topics)
        self.tab_vpc.cmb_pc.clear()
        self.tab_vpc.cmb_pc.addItems(pc_topics)
        if image_topics:
            self.tab_vpc.cmb_video.setCurrentIndex(0)
            self._change_video_topic(image_topics[0])
        if pc_topics:
            self.tab_vpc.cmb_pc.setCurrentIndex(0)
            self._change_pc_topic(pc_topics[0])

        progress.set_bar_steps(3)
        if not progress.advance("Erzeuge DataFrames …"):
            log.debug("Aborted during DataFrame generation")
            progress.accept()
            return
        try:
            self._build_dfs()
            log.debug("DataFrames built")
        except Exception:
            log.exception("Error while building DataFrames")
            progress.accept()
            return

        if not progress.advance("Vorverarbeitung …"):
            log.debug("Aborted during preprocessing")
            progress.accept()
            return
        try:
            self._preprocess_all()
            log.debug("Preprocessing done")
        except Exception:
            log.exception("Error during preprocessing")
            progress.accept()
            return

        if not progress.advance("Erstelle Plots …"):
            log.debug("Aborted before plotting")
            progress.accept()
            return
        self._set_defaults()
        try:
            self._draw_plots()
            log.debug("Plots drawn")
        except Exception:
            log.exception("Error while drawing plots")
            progress.accept()
            return

        if not progress.advance("Erstelle Karte …"):
            log.debug("Aborted before map creation")
            progress.accept()
            return
        try:
            self._draw_map()
            log.debug("Map drawn")
        except Exception:
            log.exception("Error while drawing map")
            progress.accept()
            return
        progress.accept()

        self.act_topics.setEnabled(True)
        self.act_verify.setEnabled(True)
        self.act_export.setEnabled(True)
        self.act_check.setEnabled(True)

    # ------------------------------------------------------------------ DataFrame
    def _build_dfs(self) -> None:
        log.debug("Building DataFrames")
        for t, samps in self.samples.items():
            if not samps:
                continue
            ta = np.array([s.time_abs for s in samps])
            if self.t0 is None:
                self.t0 = ta[0]
            tr = np.maximum(ta - self.t0, 0)
            df = pd.DataFrame({
                "time": tr, "time_abs": ta,
                "accel_x": [s.lin_acc[0] for s in samps],
                "accel_y": [s.lin_acc[1] for s in samps],
                "accel_z": [s.lin_acc[2] for s in samps],
                "label_id": np.full_like(tr, UNKNOWN_ID, int),
                "label_name": [UNKNOWN_NAME] * len(tr),
            })
            self.dfs[t] = df
        log.debug("DataFrames created: %s", list(self.dfs))

    # ------------------------------------------------------------------ Preprocess
    def _preprocess_all(self) -> None:
        log.debug("Preprocessing all topics")
        self.iso_metrics.clear()
        arglist = []
        for topic, df in self.dfs.items():
            samps = self.samples[topic]
            acc = np.asarray([s.lin_acc for s in samps], dtype=np.float32)
            ori = np.asarray([
                [s.msg.orientation.x, s.msg.orientation.y, s.msg.orientation.z, s.msg.orientation.w]
                for s in samps
            ], dtype=np.float32)
            arglist.append(
                (
                    topic,
                    df.copy(),
                    acc,
                    ori,
                    self._gps_df,
                    self.rot_mode,
                    self.mount_overrides,
                    self.iso_comfort,
                    self.peak_threshold,
                    self.peak_distance,
                    self.use_max_peak,
                )
            )

        with ProcessPoolExecutor() as pool:
            for topic, df_new, iso in pool.map(_preprocess_single, arglist):
                self.dfs[topic] = df_new
                self.iso_metrics[topic] = iso
        log.debug("Preprocessing finished")

    def _resolve_rotation(self, topic: str, df: pd.DataFrame) -> np.ndarray | None:
        auto = auto_vehicle_frame(df, self._gps_df)
        ov = self.mount_overrides.get(topic)

        if self.rot_mode == RotMode.OVERRIDE_FIRST:
            if ov is not None:
                rot = ov
            else:
                rot = auto
        elif self.rot_mode == RotMode.AUTO_FIRST:
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
            "/ouster_rear/imu",
        ]
        available = self.available_topics or list(self.dfs)
        if not available:
            self.active_topics = []
            return

        sel = [p for p in pref if p in available]
        for t in available:
            if t not in sel:
                sel.append(t)

        self.active_topics = sel

    # ------------------------------------------------------------------ Plots
    def _draw_plots(self, verify: bool = False) -> None:
        log.debug("Drawing plots (verify=%s)", verify)
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
                    ax.plot(df["time"], df["accel_x"], label="accel_x", color="tab:blue")
                if self.act_show_y.isChecked():
                    ax.plot(df["time"], df["accel_y"], label="accel_y", color="tab:orange")
                if self.act_show_z.isChecked():
                    ax.plot(df["time"], df["accel_z"], label="accel_z", color="tab:green")
            if self.act_show_corr.isChecked() and "accel_corr_x" in df.columns:
                if self.act_show_x.isChecked():
                    ax.plot(df["time"], df["accel_corr_x"], label="accel_corr_x", color="tab:blue", alpha=0.8, ls="--")
                if self.act_show_y.isChecked():
                    ax.plot(df["time"], df["accel_corr_y"], label="accel_corr_y", color="tab:orange", alpha=0.8, ls="--")
                if self.act_show_z.isChecked():
                    ax.plot(df["time"], df["accel_corr_z"], label="accel_corr_z", color="tab:green", alpha=0.8, ls="--")
            if self.act_show_veh.isChecked() and {"accel_veh_x", "accel_veh_y", "accel_veh_z"}.issubset(df.columns):
                if self.act_show_x.isChecked():
                    ax.plot(df["time"], df["accel_veh_x"], label="accel_veh_x", color="tab:blue", alpha=0.6, ls=":")
                if self.act_show_y.isChecked():
                    ax.plot(df["time"], df["accel_veh_y"], label="accel_veh_y", color="tab:orange", alpha=0.6, ls=":")
                if self.act_show_z.isChecked():
                    ax.plot(df["time"], df["accel_veh_z"], label="accel_veh_z", color="tab:green", alpha=0.6, ls=":")
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
                segs = self.label_patches_track.get(topic, [])
                if segs:
                    for s, e, lname in segs:
                        color = ANOMALY_TYPES[lname]["color"]
                        ax_tr.axvspan(s, e, color=color, alpha=.6, label=lname)
                else:
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
        log.debug("Plots updated")
        if getattr(self, "tabs", None) and self.tabs.currentIndex() == 1:
            self._draw_map()

    def _span_selected(self, topic: str, xmin: float, xmax: float) -> None:
        self.current_span[topic] = (xmin, xmax)
        self.last_selected_topic = topic

    def _restore_labels(self, ax, topic: str) -> None:
        df = self.dfs[topic]
        self.label_patches[topic] = []
        self.label_patches_track[topic] = []
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
                    self.label_patches_track[topic].append((start, prev_t, last))
                start = t
                last = name
            prev_t = t
        if last != UNKNOWN_NAME and start is not None:
            color = ANOMALY_TYPES[last]["color"]
            patch = ax.axvspan(start, prev_t, alpha=.2, color=color, label=last)
            self.label_patches[topic].append((start, prev_t, patch))
            self.label_patches_track[topic].append((start, prev_t, last))

    # ------------------------------------------------------------------ Maus
    def _mouse_press(self, ev) -> None:
        if ev.inaxes not in self.ax_topic:
            return
        topic = self.ax_topic[ev.inaxes]

        if ev.button == 1:
            peaks = self.iso_metrics.get(topic, {}).get("peaks", [])
            if peaks:
                df = self.dfs[topic]
                times = df.loc[peaks, "time"].to_numpy()
                idx = int(np.argmin(np.abs(times - ev.xdata)))
                if abs(times[idx] - ev.xdata) <= 0.2:
                    self._show_peak_video(topic, times[idx])
                    return

        if ev.button != 3:
            return
        if topic not in self.current_span:
            return
        xmin, xmax = self.current_span[topic]
        if xmax <= xmin:
            return
        lname = self.cmb.currentData()
        self.undo.push(AddLabelCmd(self, topic, xmin, xmax, lname))

    def _tab_changed(self, idx: int) -> None:
        if idx == 1:
            self._draw_map()
        elif idx == 2:
            if not self.tab_vpc.video_frames:
                self.tab_vpc.show_pointcloud_placeholder("No data loaded")

    def _change_video_topic(self, t: str) -> None:
        self.video_topic = t if t else None
        vframes = self.video_frames_by_topic.get(t, [])
        pframes = self.pc_frames_by_topic.get(self.pc_topic, [])
        self.tab_vpc.load_arrays(vframes, pframes)

    def _change_pc_topic(self, t: str) -> None:
        self.pc_topic = t if t else None
        vframes = self.video_frames_by_topic.get(self.video_topic, [])
        pframes = self.pc_frames_by_topic.get(t, [])
        self.tab_vpc.load_arrays(vframes, pframes)

    def _show_peak_video(self, topic: str, t_peak: float) -> None:
        vid_topic = self.video_topic or ""
        pc_topic = self.pc_topic or ""
        vframes = self.video_frames_by_topic.get(vid_topic, [])
        pframes = self.pc_frames_by_topic.get(pc_topic, [])
        vtimes = self.video_times_by_topic.get(vid_topic, [])
        ptimes = self.pc_times_by_topic.get(pc_topic, [])
        if not vframes and not pframes:
            return

        pre = self.tab_vpc.spn_pre.value()
        post = self.tab_vpc.spn_post.value()
        t0 = self.t0 or 0.0
        start = t_peak - pre
        end = t_peak + post

        if vtimes:
            tr = np.array(vtimes) - t0
            i0 = int(np.searchsorted(tr, start, "left"))
            i1 = int(np.searchsorted(tr, end, "right"))
            vframes = vframes[i0:i1]
        if ptimes:
            trp = np.array(ptimes) - t0
            i0 = int(np.searchsorted(trp, start, "left"))
            i1 = int(np.searchsorted(trp, end, "right"))
            pframes = pframes[i0:i1]

        self.tab_vpc.load_arrays(vframes, pframes)
        self.tabs.setCurrentIndex(2)
        self.tab_vpc.play()

    def _draw_map(self) -> None:
        log.debug("Drawing map")
        if self._gps_df is None or not self.active_topics:
            if QWebEngineView is None:
                self.fig_map.clear()
                self.canvas_map.draw_idle()
            else:
                self.web_map.setHtml("<p>No GPS data.</p>")
            return

        topic = self.active_topics[0]
        df = self.dfs[topic]

        # interpolate AWV to GPS timestamps
        gps = self._gps_df.copy()
        gps["awv"] = np.interp(gps["time"], df["time_abs"], df.get("awv", pd.Series(np.zeros(len(df)))) )

        # optional peak markers
        gps["peak"] = False
        if self.act_show_peaks.isChecked():
            peaks = self.iso_metrics.get(topic, {}).get("peaks", [])
            if len(peaks):
                peak_times = df.loc[peaks, "time_abs"].to_numpy()
                tol = 0.1
                gps["peak"] = gps["time"].apply(lambda t: bool(np.any(np.abs(t - peak_times) <= tol)))

        if QWebEngineView is None:
            # simple matplotlib fallback
            self.fig_map.clear()
            ax = self.fig_map.add_subplot(111)
            sc = ax.scatter(gps["lon"], gps["lat"], c=gps["awv"], cmap="plasma", s=5)
            self.fig_map.colorbar(sc, ax=ax, label="awv")
            if gps["peak"].any():
                ax.scatter(gps.loc[gps["peak"], "lon"], gps.loc[gps["peak"], "lat"],
                           facecolors="none", edgecolors="black", s=40)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("GPS Track")
            self.canvas_map.draw_idle()
        else:
            try:
                import folium
                from PyQt5.QtCore import QUrl
                import tempfile, os
            except Exception:
                self.web_map.setHtml("<p>Folium not available.</p>")
                return

            lat0 = gps["lat"].mean()
            lon0 = gps["lon"].mean()
            fmap = folium.Map(location=[lat0, lon0], zoom_start=16, max_zoom=30, min_zoom=10)

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
                        color=c_for(row.awv), fill=True
                    ).add_to(fmap)

            map_file = os.path.join(tempfile.gettempdir(), "analysis_map.html")
            fmap.save(map_file)
            gps.to_csv(map_file.replace(".html", "_gps.csv"), index=False)
            self.web_map.load(QUrl.fromLocalFile(map_file))
        log.debug("Map updated")

    def _assign_label(self, topic: str, xmin: float, xmax: float,
                      lname: str | None = None) -> None:
        df = self.dfs[topic]
        if lname is None:
            lname = self.cmb.currentData()
        lid = self.LABEL_IDS[lname]
        color = ANOMALY_TYPES[lname]["color"]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        df.loc[mask, ["label_id", "label_name"]] = [lid, lname]

        ax = next(a for a, t in self.ax_topic.items() if t == topic)
        patch = ax.axvspan(xmin, xmax, alpha=.2, color=color, label=lname)
        self.label_patches.setdefault(topic, []).append((xmin, xmax, patch))
        self.label_patches_track.setdefault(topic, []).append((xmin, xmax, lname))
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", ncol=2)
        self.canvas.draw_idle()

    def _delete_label_range(self, topic: str, xmin: float, xmax: float) -> None:
        df = self.dfs[topic]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        df.loc[mask, ["label_id", "label_name"]] = [UNKNOWN_ID, UNKNOWN_NAME]

        # remove all existing patches for this topic and rebuild from DataFrame
        if topic in self.label_patches:
            for _, _, patch in self.label_patches[topic]:
                patch.remove()
            self.label_patches[topic] = []
        if topic in self.label_patches_track:
            self.label_patches_track[topic] = []
        ax = next(a for a, t in self.ax_topic.items() if t == topic)
        self._restore_labels(ax, topic)
        self.canvas.draw_idle()

    def _button_add_label(self) -> None:
        t = self.last_selected_topic
        if not t or t not in self.current_span:
            return
        xmin, xmax = self.current_span[t]
        if xmax <= xmin:
            return
        lname = self.cmb.currentData()
        self.undo.push(AddLabelCmd(self, t, xmin, xmax, lname))

    def _button_edit_label(self) -> None:
        t = self.last_selected_topic
        if not t or t not in self.current_span:
            return
        xmin, xmax = self.current_span[t]
        if xmax <= xmin:
            return
        lname = self.cmb.currentData()
        self.undo.push(EditLabelCmd(self, t, xmin, xmax, lname))

    def _button_delete_label(self) -> None:
        t = self.last_selected_topic
        if not t or t not in self.current_span:
            return
        xmin, xmax = self.current_span[t]
        self.undo.push(DeleteLabelCmd(self, t, xmin, xmax))

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
                rot_ok = {"accel_veh_x", "accel_veh_y", "accel_veh_z"}.issubset(df.columns) or \
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

    def _open_label_manager(self) -> None:
        dlg = LabelManagerDialog(self.dfs, self)
        if dlg.exec() != QDialog.Accepted:
            return
        result = dlg.result()
        for topic, segs in result.items():
            df = self.dfs[topic]
            df["label_id"] = UNKNOWN_ID
            df["label_name"] = UNKNOWN_NAME
            for lbl, s, e in segs:
                lid = self.LABEL_IDS.get(lbl, UNKNOWN_ID)
                mask = (df["time"] >= s) & (df["time"] <= e)
                df.loc[mask, ["label_id", "label_name"]] = [lid, lbl]
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
