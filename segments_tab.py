from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QGridLayout
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # Matplotlib < 3.8
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from labels import ANOMALY_TYPES


class SegmentsTab(QWidget):
    """Browse labeled segment CSVs and peak images."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.root_folder: Optional[Path] = None
        self.label_folders: Dict[str, List[Path]] = {}
        self.current_label: str = ""
        self.current_index: int = 0

        self._build_ui()
        self._connect_signals()
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)

        # folder selection
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel("-")
        hl.addWidget(self.lbl_folder, 1)
        self.btn_browse = QPushButton("Browse…")
        hl.addWidget(self.btn_browse)
        vbox.addLayout(hl)

        # label folder + segment label selection
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Label folder:"))
        self.cmb_folder = QComboBox()
        hl2.addWidget(self.cmb_folder)
        hl2.addWidget(QLabel("Segment label:"))
        self.cmb_label = QComboBox()
        self.cmb_label.addItems(list(ANOMALY_TYPES))
        hl2.addWidget(self.cmb_label)
        vbox.addLayout(hl2)

        # plot
        self.fig = Figure(figsize=(8, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)

        # peak images
        grid = QGridLayout()
        self.lbl_peaks: List[QLabel] = []
        for i in range(6):
            lbl = QLabel("No image")
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, i // 3, i % 3)
            self.lbl_peaks.append(lbl)
        vbox.addLayout(grid)

    def _connect_signals(self) -> None:
        self.btn_browse.clicked.connect(self._browse)
        self.cmb_folder.currentTextChanged.connect(self._folder_changed)

    # ------------------------------------------------------------------ folder logic
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select segments folder", str(Path.home())
        )
        if path:
            self.load_folder(Path(path))

    def load_folder(self, folder: Path) -> None:
        self.root_folder = folder
        self.lbl_folder.setText(str(folder))

        self.label_folders.clear()
        self.cmb_folder.blockSignals(True)
        self.cmb_folder.clear()
        for sub in sorted(folder.iterdir()):
            if sub.is_dir():
                csvs = sorted([c for c in sub.glob("*.csv")])
                if csvs:
                    self.label_folders[sub.name] = csvs
                    self.cmb_folder.addItem(sub.name)
        self.cmb_folder.blockSignals(False)

        if self.label_folders:
            self.cmb_folder.setCurrentIndex(0)
            self._folder_changed(self.cmb_folder.currentText())

    def _folder_changed(self, label: str) -> None:
        self.current_label = label
        self.current_index = 0
        self._load_current()

    # ------------------------------------------------------------------ segment logic
    def _load_current(self) -> None:
        for lbl in self.lbl_peaks:
            lbl.clear()
            lbl.setText("No image")
        self.fig.clf()

        csvs = self.label_folders.get(self.current_label)
        if not csvs:
            self.canvas.draw_idle()
            return

        csv = csvs[self.current_index]
        try:
            df = pd.read_csv(csv)
        except Exception:
            self.canvas.draw_idle()
            return

        self.cmb_label.setCurrentText(self.current_label)

        ax = self.fig.add_subplot(111)
        for col in ("accel_x", "accel_y", "accel_z"):
            if col in df.columns:
                ax.plot(df["time"], df[col], label=col)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("m/s²")
        ax.legend(fontsize="x-small")
        self.canvas.draw_idle()

        seg_id = csv.stem.split("_")[-1]
        peak_dir = csv.parent / "peaks" / seg_id
        if peak_dir.is_dir():
            for lbl, img in zip(self.lbl_peaks, sorted(peak_dir.glob("*.png"))):
                pix = QPixmap(str(img))
                if lbl.width() and lbl.height():
                    pix = pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pix)

    # ------------------------------------------------------------------ navigation
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Right:
            self._next()
            return
        if event.key() == Qt.Key_Left:
            self._prev()
            return
        super().keyPressEvent(event)

    def _next(self) -> None:
        csvs = self.label_folders.get(self.current_label)
        if csvs and self.current_index + 1 < len(csvs):
            self.current_index += 1
            self._load_current()

    def _prev(self) -> None:
        csvs = self.label_folders.get(self.current_label)
        if csvs and self.current_index > 0:
            self.current_index -= 1
            self._load_current()
