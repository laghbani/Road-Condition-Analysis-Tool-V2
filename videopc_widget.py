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
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox,
    QGridLayout, QComboBox, QDialog, QDialogButtonBox, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
import numpy as np


class VideoPointCloudPair(QWidget):
    """A single video/point cloud pair with topic selection."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.cmb_video = QComboBox()
        self.cmb_pc = QComboBox()
        top.addWidget(QLabel("Video:"))
        top.addWidget(self.cmb_video)
        top.addWidget(QLabel("PC:"))
        top.addWidget(self.cmb_pc)
        layout.addLayout(top)

        grid = QGridLayout()
        self.btn_video_report = QPushButton("Add Video to Report")
        self.btn_pc_report = QPushButton("Add PC to Report")
        grid.addWidget(self.btn_video_report, 0, 0)
        grid.addWidget(self.btn_pc_report, 0, 1)

        self.lbl_video = QLabel("Video Frame")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(320, 240)
        grid.addWidget(self.lbl_video, 1, 0)

        self.lbl_pc = QLabel("Point Cloud View")
        self.lbl_pc.setAlignment(Qt.AlignCenter)
        self.lbl_pc.setMinimumSize(320, 240)
        grid.addWidget(self.lbl_pc, 1, 1)

        layout.addLayout(grid)

    # -------------------------------------------------------------- helpers
    def show_video_frame(self, img: QImage) -> None:
        self.lbl_video.setPixmap(QPixmap.fromImage(img))

    def show_pointcloud_image(self, img: QImage) -> None:
        self.lbl_pc.setPixmap(QPixmap.fromImage(img))

    @property
    def video_topic(self) -> str:
        return self.cmb_video.currentText()

    @video_topic.setter
    def video_topic(self, val: str) -> None:
        idx = self.cmb_video.findText(val)
        if idx >= 0:
            self.cmb_video.setCurrentIndex(idx)

    @property
    def pc_topic(self) -> str:
        return self.cmb_pc.currentText()

    @pc_topic.setter
    def pc_topic(self, val: str) -> None:
        idx = self.cmb_pc.findText(val)
        if idx >= 0:
            self.cmb_pc.setCurrentIndex(idx)


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
        self.pairs = [VideoPointCloudPair(), VideoPointCloudPair()]
        grid = QHBoxLayout()
        for p in self.pairs:
            grid.addWidget(p, stretch=1)
        vbox.addLayout(grid, stretch=1)

        self.frames: list[QImage] = []
        self.pc_frames: list[QImage] = []
        self.sync_index = 0
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._timer_tick)

        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_replay.clicked.connect(self.replay)

        self.roi_mode = False
        self.btn_toggle_roi.clicked.connect(self._toggle_roi)

        self.available_topics: list[str] = []

    # ------------------------------------------------------------------ actions
    def _toggle_roi(self) -> None:
        self.roi_mode = not self.roi_mode
        self.btn_toggle_roi.setChecked(self.roi_mode)

    # -------------------------------------------------------------- playback
    def load_demo_peak(self, topic: str, t_peak: float) -> None:
        pre = self.spn_pre.value()
        post = self.spn_post.value()
        n = int((pre + post) / 0.1) + 1
        self.frames = []
        self.pc_frames = []
        for i in range(n):
            img = QImage(320, 240, QImage.Format_RGB32)
            img.fill(Qt.white)
            painter = QPainter(img)
            painter.drawText(10, 20, f"{topic} {t_peak:.2f}s #{i}")
            painter.end()
            self.frames.append(img)

            pc = QImage(320, 240, QImage.Format_RGB32)
            color = QColor.fromHsv((i * 30) % 360, 255, 200)
            pc.fill(color)
            self.pc_frames.append(pc)

        self.sync_index = 0
        if self.frames:
            self.show_video_frame(self.frames[0])
            self.show_pointcloud_image(self.pc_frames[0])

    def play(self) -> None:
        if self.frames:
            self.timer.start()

    def pause(self) -> None:
        self.timer.stop()

    def replay(self) -> None:
        self.sync_index = 0
        if self.frames:
            self.show_video_frame(self.frames[0])
            self.show_pointcloud_image(self.pc_frames[0])
        self.timer.start()

    def _timer_tick(self) -> None:
        if self.sync_index >= len(self.frames):
            self.timer.stop()
            return
        img = self.frames[self.sync_index]
        pc = self.pc_frames[self.sync_index]
        self.show_video_frame(img)
        self.show_pointcloud_image(pc)
        self.sync_index += 1

    # ------------------------------------------------------------------ helpers
    def show_video_frame(self, img: QImage) -> None:
        for p in self.pairs:
            p.show_video_frame(img)

    def show_pointcloud_placeholder(self, text: str) -> None:
        """Display placeholder text in the point cloud view."""
        for p in self.pairs:
            p.lbl_pc.setText(text)

    def show_pointcloud_image(self, img: QImage) -> None:
        for p in self.pairs:
            p.show_pointcloud_image(img)

    def save_video_screenshot(self, path: Path) -> None:
        pix = self.pairs[0].lbl_video.grab()
        path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(path))

    def save_pc_screenshot(self, path: Path) -> None:
        pix = self.pairs[0].lbl_pc.grab()
        path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(path))

    # -------------------------------------------------------------- topics
    def set_available_topics(self, topics: list[str]) -> None:
        self.available_topics = topics
        for p in self.pairs:
            p.cmb_video.clear()
            p.cmb_pc.clear()
            p.cmb_video.addItems(topics)
            p.cmb_pc.addItems(topics)

    def open_topic_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Topics")
        form = QFormLayout(dlg)
        cmb_v = []
        cmb_p = []
        for i, p in enumerate(self.pairs):
            cv = QComboBox()
            cp = QComboBox()
            cv.addItems(self.available_topics)
            cp.addItems(self.available_topics)
            cv.setCurrentText(p.video_topic)
            cp.setCurrentText(p.pc_topic)
            form.addRow(f"Pair {i+1} video", cv)
            form.addRow(f"Pair {i+1} PC", cp)
            cmb_v.append(cv)
            cmb_p.append(cp)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec_() == QDialog.Accepted:
            for p, cv, cp in zip(self.pairs, cmb_v, cmb_p):
                p.video_topic = cv.currentText()
                p.pc_topic = cp.currentText()
