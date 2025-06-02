# -*- coding: utf-8 -*-
"""Simple placeholder widgets for synchronized video and point cloud playback.

This module provides a ``VideoPointCloudTab`` widget that contains two columns
for a video frame and a point cloud view. The layout roughly matches the
requested design but does not implement full 3-D rendering or synchronization.
It merely demonstrates how the GUI could be structured.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


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
        self.btn_video_report = QPushButton("Add Video to Report")
        vcol.addWidget(self.btn_video_report)
        self.lbl_video = QLabel("Video Frame")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(320, 240)
        vcol.addWidget(self.lbl_video, stretch=1)
        hbox.addLayout(vcol, stretch=1)

        # ------------------------------ Point cloud column
        pcol = QVBoxLayout()
        self.btn_pc_report = QPushButton("Add PC to Report")
        pcol.addWidget(self.btn_pc_report)
        self.lbl_pc = QLabel("Point Cloud View")
        self.lbl_pc.setAlignment(Qt.AlignCenter)
        self.lbl_pc.setMinimumSize(320, 240)
        pcol.addWidget(self.lbl_pc, stretch=1)
        hbox.addLayout(pcol, stretch=1)

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
        self.lbl_pc.setText(text)

    def save_video_screenshot(self, path: Path) -> None:
        pix = self.lbl_video.grab()
        path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(path))

    def save_pc_screenshot(self, path: Path) -> None:
        pix = self.lbl_pc.grab()
        path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(path))
