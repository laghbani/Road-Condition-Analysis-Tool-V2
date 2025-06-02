# -*- coding: utf-8 -*-
"""Video and point cloud playback widget.

The :class:`VideoPointCloudTab` now supports displaying video frames from
NumPy arrays via :func:`show_frame` and rendering 3-D point clouds with
``pyqtgraph``.  Point size can be adjusted interactively.
"""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox, QComboBox, QSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

# ─────────────────────────────────────────────────────────────────────────────
# ▼▼▼  NEUER BLOCK – direkt hinter den bisherigen QWidget-Imports einfügen  ▼▼▼
# ─────────────────────────────────────────────────────────────────────────────
from scipy.spatial.transform import Rotation as R   # pip install scipy
# ─────────────────────────────────────────────────────────────────────────────
# ▲▲▲  NEUER BLOCK ENDE                                                     ▲▲▲
# ─────────────────────────────────────────────────────────────────────────────

import cv2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

import numpy as np

import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem


def _downsample(iter_pts, step: int = 8):
    """Yield every *step*-th point from an iterator."""
    for k, p in enumerate(iter_pts):
        if k % step == 0:
            yield p


def _pc_to_xyz(pc_msg: PointCloud2, step: int = 1) -> np.ndarray:
    """Alle Punkte → Nx3-float32-Array (kein Decimation, wenn step=1)."""
    # 1) Rohpunkte als strukturierte Records einlesen
    rec = np.fromiter(
        pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        count=-1,  # Disable pre-allocation when size unknown
    )

    # 2) Optionaler Down-Sampler
    if step > 1:
        rec = rec[::step]

    # 3) Spalten sauber stapeln ⇒ echtes (N,3)-float32-Array
    pts = np.column_stack((rec["x"], rec["y"], rec["z"])).astype(np.float32, copy=False)
    return pts


class VideoPointCloudTab(QWidget):
    """Tab with video player and point cloud viewer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        vbox = QVBoxLayout(self)

        # -------------------------------------------------- Control bar
        ctrl = QHBoxLayout()

        # Zeitfenster
        for cap, box in (("Pre [s]:", "spn_pre"), ("Post [s]:", "spn_post")):
            ctrl.addWidget(QLabel(cap))
            setattr(self, box, QDoubleSpinBox())
            sb: QDoubleSpinBox = getattr(self, box)
            sb.setRange(0.0, 10.0); sb.setDecimals(1); sb.setValue(2.0)
            ctrl.addWidget(sb)

        # ▶️/⏸ Steuerung
        self.btn_play, self.btn_pause, self.btn_replay = (
            QPushButton(t) for t in ("Play", "Pause", "Replay")
        )
        for b in (self.btn_play, self.btn_pause, self.btn_replay):
            ctrl.addWidget(b)

        # Punktgröße
        ctrl.addWidget(QLabel("PtSize:"))
        self.spn_size = QDoubleSpinBox(); self.spn_size.setRange(1, 20); self.spn_size.setValue(5)
        self.spn_size.valueChanged.connect(self.update_point_size)
        ctrl.addWidget(self.spn_size)

        # FPS
        ctrl.addWidget(QLabel("FPS:"))
        self.spn_fps = QSpinBox(); self.spn_fps.setRange(1, 60); self.spn_fps.setValue(10)
        ctrl.addWidget(self.spn_fps)

        # ↓↓↓ NEU ↓↓↓ --- Parameter fürs Point-Processing ---
        ctrl.addWidget(QLabel("Decimate:"))
        self.spn_dec = QSpinBox(); self.spn_dec.setRange(1, 16); self.spn_dec.setValue(1)
        ctrl.addWidget(self.spn_dec)

        self.chk_show_ground = QCheckBox("Only ground"); ctrl.addWidget(self.chk_show_ground)

        ctrl.addWidget(QLabel("Ground ±[cm]:"))
        self.spn_gth = QSpinBox(); self.spn_gth.setRange(1, 50); self.spn_gth.setValue(10)
        ctrl.addWidget(self.spn_gth)

        # Yaw / Pitch / Roll für Fahrzeug-Frame
        for cap, box in (("Yaw°:", "spn_yaw"), ("Pitch°:", "spn_pitch"), ("Roll°:", "spn_roll")):
            ctrl.addWidget(QLabel(cap))
            setattr(self, box, QDoubleSpinBox())
            sb: QDoubleSpinBox = getattr(self, box)
            sb.setRange(-180, 180); sb.setDecimals(1); sb.setValue(0.0)
            ctrl.addWidget(sb)

        # Boden/Anomalie-Anzeige
        self.btn_detect = QPushButton("Detect"); ctrl.addWidget(self.btn_detect)

        ctrl.addStretch()
        vbox.addLayout(ctrl)

        # -------------------------------------------------- Main area
        hbox = QHBoxLayout()
        vbox.addLayout(hbox, stretch=1)

        # ------------------------------ Video column
        vcol = QVBoxLayout()
        self.lbl_video = QLabel("Video Frame")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(320, 240)
        vcol.addWidget(self.lbl_video, stretch=1)
        self.cmb_video = QComboBox()
        vcol.addWidget(self.cmb_video)
        hbox.addLayout(vcol, stretch=1)

        # ------------------------------ Point cloud column
        pcol = QVBoxLayout()
        self.gl_view = GLViewWidget()
        self.gl_view.setMinimumSize(320, 240)
        self.gl_view.opts["distance"] = 5
        self.gl_view.opts["projection"] = "perspective"
        pcol.addWidget(self.gl_view, stretch=1)
        self.lbl_placeholder = QLabel("Point Cloud View")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.hide()
        pcol.addWidget(self.lbl_placeholder, stretch=1)
        self.cmb_pc = QComboBox()
        pcol.addWidget(self.cmb_pc)
        hbox.addLayout(pcol, stretch=1)

        self.video_frames: list[QImage] = []
        self.pc_frames: list[QImage] = []
        self.img_arrays: list = []
        self.pc_arrays: list = []
        self.scatter_item: GLScatterPlotItem | None = None
        self._last_pts: np.ndarray | None = None
        self._last_cols: np.ndarray | None = None
        self.pc_times: list[float] = []
        self._acc_window_s = 2.0
        self.spn_pre.valueChanged.connect(lambda v: setattr(self, "_acc_window_s", v))
        # Detect-Button
        self.btn_detect.clicked.connect(self._detect_anomalies)
        self.point_size = self.spn_size.value()
        self.sync_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_replay.clicked.connect(self.replay)

    # ------------------------------------------------------------------ helpers
    def show_video_frame(self, img: QImage) -> None:
        self.lbl_video.setPixmap(QPixmap.fromImage(img))

    def show_pointcloud_placeholder(self, text: str) -> None:
        """Display placeholder text in the point cloud view."""
        self.gl_view.setVisible(False)
        self.lbl_placeholder.setText(text)
        self.lbl_placeholder.setVisible(True)

    # ------------------------------ playback ------------------------------
    def load_frames(self, video: list[QImage], pc: list[QImage]) -> None:
        self.video_frames = video
        self.pc_frames = pc
        self.sync_index = 0
        if video:
            self.show_video_frame(video[0])
        if pc:
            self.show_pc_frame(pc[0])

    def load_arrays(self, images: list, pcs: list, pc_times: list | None = None) -> None:
        """Load raw data for video frames and point clouds."""
        self.img_arrays = images
        self.pc_arrays = pcs
        self.pc_times = pc_times or []
        self.sync_index = 0
        if images:
            self.show_frame(0)
        if pcs:
            # first accumulation draw
            self.draw_scatter([pcs[0]])
        self.play()

    def play(self) -> None:
        """Start playback if any frames are available."""
        if not (self.video_frames or self.img_arrays):
            return
        interval = int(1000 / max(1, self.spn_fps.value()))
        self.timer.start(interval)

    def pause(self) -> None:
        self.timer.stop()

    def replay(self) -> None:
        self.sync_index = 0
        if self.video_frames:
            self.show_video_frame(self.video_frames[0])
        if self.pc_frames:
            self.show_pc_frame(self.pc_frames[0])
        self.play()

    def show_frame(self, i: int) -> None:
        if not self.img_arrays:
            return
        i = max(0, min(i, len(self.img_arrays) - 1))
        raw = self.img_arrays[i]
        if isinstance(raw, Image):
            arr = CvBridge().imgmsg_to_cv2(raw, desired_encoding="bgr8")
        elif isinstance(raw, (bytes, memoryview)):
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return
        else:
            arr = raw
        if arr.ndim == 2:
            h, w = arr.shape
            fmt = QImage.Format_Grayscale8
            bytes_per_line = w
        else:
            h, w, ch = arr.shape
            fmt = getattr(QImage, "Format_BGR888", QImage.Format_RGB888) if ch == 3 else QImage.Format_RGBA8888
            if ch == 3 and fmt == QImage.Format_BGR888:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                fmt = QImage.Format_RGB888
            elif ch == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
        qimg = QImage(arr.tobytes(), w, h, bytes_per_line, fmt)
        qimg = qimg.copy()
        self.show_video_frame(qimg)

    def draw_scatter(self, pts_raw) -> None:
        """Render single or multiple point clouds."""
        # -- 0. Eingangsdaten → numpy Nx3 ----------------------------------
        pts_list = []
        if isinstance(pts_raw, list):
            items = pts_raw
        else:
            items = [pts_raw]
        for item in items:
            if isinstance(item, np.ndarray):
                arr = item
            elif isinstance(item, PointCloud2):
                arr = _pc_to_xyz(item, step=max(1, self.spn_dec.value()))
            else:
                arr = np.asarray(item, dtype=np.float32)
            if arr.size:
                pts_list.append(arr)
        if not pts_list:
            return
        pts = np.vstack(pts_list)

        # -- 1. Transform in Fahrzeug-Frame ---------------------------------
        stack = self._to_vehicle(pts)

        # -- 2. Farbe/Z-Kodierung -----------------------------------------
        z = stack[:, 2]
        zmin, rng = float(z.min()), float(max(z.ptp(), 1e-3))
        norm = (z - zmin) / rng
        colors = np.column_stack((norm, np.zeros_like(norm), 1 - norm,
                                  np.ones_like(norm)))
        if self.chk_show_ground.isChecked():
            mask_ground = np.abs(stack[:, 2]) <= self.spn_gth.value() / 100.0
            stack = stack[mask_ground]
            colors = colors[mask_ground]
        self._last_pts, self._last_cols = stack, colors

        # -- 3. Plot --------------------------------------------------------
        self.gl_view.setVisible(True); self.lbl_placeholder.hide()
        if self.scatter_item is None:
            self.scatter_item = GLScatterPlotItem(pos=stack, color=colors,
                                                  size=self.point_size)
            self.gl_view.addItem(self.scatter_item)
        else:
            self.scatter_item.setData(pos=stack, color=colors,
                                      size=self.point_size)

    def update_point_size(self, val: float) -> None:
        self.point_size = val
        if self.scatter_item is not None:
            pts = self._last_pts if self._last_pts is not None else self.scatter_item.pos
            cols = self._last_cols if self._last_cols is not None else self.scatter_item.color
            self.scatter_item.setData(pos=pts, color=cols, size=val)

    def _next_frame(self) -> None:
        if (
            self.sync_index >= len(self.video_frames)
            and self.sync_index >= len(self.img_arrays)
        ):
            self.pause()
            return

        if self.sync_index < len(self.video_frames):
            self.show_video_frame(self.video_frames[self.sync_index])
        elif self.sync_index < len(self.img_arrays):
            self.show_frame(self.sync_index)

        if self.pc_frames:
            idx = min(self.sync_index, len(self.pc_frames) - 1)
            self.show_pc_frame(self.pc_frames[idx])
        elif self.pc_arrays:
            idx = min(self.sync_index, len(self.pc_arrays) - 1)
            if self.pc_times:
                times = np.array(self.pc_times)
                t_curr = times[idx]
                low = t_curr - self._acc_window_s
                high = t_curr + self._acc_window_s
                j0 = int(np.searchsorted(times, low, "left"))
                j1 = int(np.searchsorted(times, high, "right"))
                pcs_stack = self.pc_arrays[j0:j1]
            else:
                pcs_stack = [self.pc_arrays[idx]]
            self.draw_scatter(pcs_stack)

        self.sync_index += 1

    def show_pc_frame(self, img: QImage) -> None:
        # legacy method for QImage based frames
        self.lbl_placeholder.setVisible(False)
        self.gl_view.setVisible(False)
        self.lbl_placeholder.setPixmap(QPixmap.fromImage(img))
        self.lbl_placeholder.setVisible(True)

# ─────────────────────────────────────────────────────────────────────────────
# ▼▼▼  HELFERFUNKTIONEN GANZ UNTER IN DER KLASSE EINFÜGEN  ▼▼▼
# ─────────────────────────────────────────────────────────────────────────────
    # ---------- Koordinatentransform ---------------------------------
    def _to_vehicle(self, pts: np.ndarray) -> np.ndarray:
        """Dreht den Scan in Fahrzeug-Frame (Yaw/Pitch/Roll aus Spin-Boxen)."""
        ypr = [self.spn_yaw.value(), self.spn_pitch.value(), self.spn_roll.value()]
        Rmat = R.from_euler("zyx", ypr, degrees=True).as_matrix().astype(np.float32)
        return pts @ Rmat.T         # (N,3) · (3,3)^T

    # ---------- Anomalien-Detection ---------------------------------
    def _detect_anomalies(self):
        if self._last_pts is None:
            return
        veh = self._to_vehicle(self._last_pts)
        z_thr = self.spn_gth.value() / 100.0
        ground = np.abs(veh[:, 2]) <= z_thr
        xy = (veh[ground, :2] / 0.2).astype(int)
        cells: dict[tuple, list[float]] = {}
        for (i, j), z in zip(xy, veh[ground, 2]):
            cells.setdefault((i, j), []).append(z)
        if not cells:
            return
        spread = [np.ptp(v) for v in cells.values()]
        med = np.median(spread)
        bad = {k for k, v in cells.items() if np.ptp(v) > 2 * med}
        mask_bad = np.array([tuple(idx) in bad for idx in xy])
        cols = self._last_cols.copy()
        cols[ground] = self._last_cols[ground]
        cols[ground][mask_bad] = [1, 0, 1, 1]
        self.scatter_item.setData(pos=self._last_pts, color=cols, size=self.point_size)
# ─────────────────────────────────────────────────────────────────────────────
# ▲▲▲  ENDE NEUER FUNKTIONEN                                                 ▲▲▲
# ─────────────────────────────────────────────────────────────────────────────
