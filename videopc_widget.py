# -*- coding: utf-8 -*-
"""Video and point cloud playback widget.

The :class:`VideoPointCloudTab` now supports displaying video frames from
NumPy arrays via :func:`show_frame` and rendering 3-D point clouds with
``pyqtgraph``.  Point size can be adjusted interactively.
"""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage

import numpy as np

import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem


class VideoPointCloudTab(QWidget):
    """Tab with video player and point cloud viewer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        vbox = QVBoxLayout(self)

        # -------------------------------------------------- Control bar
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Pre-/Post-Zeit:"))
        self.spn_pre = QDoubleSpinBox()
        self.spn_pre.setRange(0.0, 10.0)
        self.spn_pre.setValue(2.0)
        ctrl.addWidget(self.spn_pre)
        self.spn_post = QDoubleSpinBox()
        self.spn_post.setRange(0.0, 10.0)
        self.spn_post.setValue(2.0)
        ctrl.addWidget(self.spn_post)

        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_replay = QPushButton("Replay")
        self.btn_toggle_roi = QPushButton("Toggle ROI")

        for b in (self.btn_play, self.btn_pause, self.btn_replay, self.btn_toggle_roi):
            ctrl.addWidget(b)
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
        hbox.addLayout(vcol, stretch=1)

        # ------------------------------ Point cloud column
        pcol = QVBoxLayout()
        self.gl_view = GLViewWidget()
        self.gl_view.setMinimumSize(320, 240)
        self.gl_view.opts["distance"] = 5
        pcol.addWidget(self.gl_view, stretch=1)
        self.lbl_placeholder = QLabel("Point Cloud View")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.hide()
        pcol.addWidget(self.lbl_placeholder, stretch=1)
        hbox.addLayout(pcol, stretch=1)

        ctrl.addWidget(QLabel("Point size:"))
        self.spn_size = QDoubleSpinBox()
        self.spn_size.setRange(1.0, 20.0)
        self.spn_size.setValue(5.0)
        self.spn_size.valueChanged.connect(self.update_point_size)
        ctrl.addWidget(self.spn_size)

        self.video_frames: list[QImage] = []
        self.pc_frames: list[QImage] = []
        self.img_arrays: list[np.ndarray] = []
        self.pc_arrays: list[np.ndarray] = []
        self.scatter_item: GLScatterPlotItem | None = None
        self.point_size = self.spn_size.value()
        self.sync_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_replay.clicked.connect(self.replay)

        self.roi_mode = False
        self.btn_toggle_roi.clicked.connect(self._toggle_roi)

    # ------------------------------------------------------------------ actions
    def _toggle_roi(self) -> None:
        self.roi_mode = not self.roi_mode
        self.btn_toggle_roi.setChecked(self.roi_mode)

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

    def load_arrays(self, images: list[np.ndarray], pcs: list[np.ndarray]) -> None:
        """Load NumPy arrays for video frames and point clouds."""
        self.img_arrays = images
        self.pc_arrays = pcs
        self.sync_index = 0
        if images:
            self.show_frame(0)
        if pcs:
            self.draw_scatter(pcs[0])

    def play(self) -> None:
        """Start playback if any frames are available."""
        if not (self.video_frames or self.img_arrays):
            return
        self.timer.start(100)

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
        arr = self.img_arrays[i]
        if arr.ndim == 2:
            h, w = arr.shape
            fmt = QImage.Format_Grayscale8
            bytes_per_line = w
        else:
            h, w, ch = arr.shape
            fmt = QImage.Format_RGB888 if ch == 3 else QImage.Format_RGBA8888
            bytes_per_line = ch * w
        qimg = QImage(arr.tobytes(), w, h, bytes_per_line, fmt)
        qimg = qimg.copy()
        self.show_video_frame(qimg)

    def draw_scatter(self, pts: np.ndarray) -> None:
        if pts.size == 0:
            return
        self.gl_view.setVisible(True)
        self.lbl_placeholder.setVisible(False)
        z = pts[:, 2]
        zmin = float(z.min())
        rng = float(z.max() - zmin) or 1.0
        norm = (z - zmin) / rng
        colors = np.column_stack((norm, np.zeros_like(norm), 1 - norm, np.ones_like(norm)))
        if self.scatter_item is None:
            self.scatter_item = GLScatterPlotItem(pos=pts, color=colors, size=self.point_size)
            self.gl_view.addItem(self.scatter_item)
        else:
            self.scatter_item.setData(pos=pts, color=colors, size=self.point_size)

    def update_point_size(self, val: float) -> None:
        self.point_size = val
        if self.scatter_item is not None:
            self.scatter_item.setData(size=val)

    def _next_frame(self) -> None:
        if self.sync_index >= max(len(self.video_frames), len(self.img_arrays)):
            self.pause()
            return
        if self.video_frames:
            self.show_video_frame(self.video_frames[self.sync_index])
        elif self.img_arrays:
            self.show_frame(self.sync_index)
        if self.pc_frames:
            idx = min(self.sync_index, len(self.pc_frames) - 1)
            self.show_pc_frame(self.pc_frames[idx])
        elif self.pc_arrays:
            idx = min(self.sync_index, len(self.pc_arrays) - 1)
            self.draw_scatter(self.pc_arrays[idx])
        self.sync_index += 1

    def show_pc_frame(self, img: QImage) -> None:
        # legacy method for QImage based frames
        self.lbl_placeholder.setVisible(False)
        self.gl_view.setVisible(False)
        self.lbl_placeholder.setPixmap(QPixmap.fromImage(img))
        self.lbl_placeholder.setVisible(True)
