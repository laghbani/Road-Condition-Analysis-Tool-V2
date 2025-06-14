from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QGridLayout
)

from labels import ANOMALY_TYPES


class SegmentsTab(QWidget):
    """Display segment CSVs with peak images."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.folder: Optional[Path] = None
        self.csvs: List[Path] = []
        self.current_index: int = 0

        self._build_ui()
        self._connect()
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)

        # ---- folder line -------------------------------------------------
        hl_path = QHBoxLayout()
        hl_path.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel("-")
        hl_path.addWidget(self.lbl_folder, 1)
        self.btn_browse = QPushButton("Browse…")
        hl_path.addWidget(self.btn_browse)
        vbox.addLayout(hl_path)

        # ---- class selection --------------------------------------------
        hl_cls = QHBoxLayout()
        hl_cls.addWidget(QLabel("Class:"))
        self.cmb_class = QComboBox()
        hl_cls.addWidget(self.cmb_class)
        vbox.addLayout(hl_cls)

        # ---- plot --------------------------------------------------------
        self.fig = Figure(figsize=(9, 4), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas)

        # ---- label selection --------------------------------------------
        hl_lbl = QHBoxLayout()
        hl_lbl.addWidget(QLabel("Label:"))
        self.cmb_label = QComboBox()
        self.cmb_label.addItems(list(ANOMALY_TYPES))
        hl_lbl.addWidget(self.cmb_label)
        hl_lbl.addStretch()
        vbox.addLayout(hl_lbl)

        # ---- peak images -------------------------------------------------
        grid = QGridLayout()
        self.lbl_peaks: List[QLabel] = []
        for i in range(6):
            lbl = QLabel("No image")
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, i // 3, i % 3)
            self.lbl_peaks.append(lbl)
        vbox.addLayout(grid)

    def _connect(self) -> None:
        self.btn_browse.clicked.connect(self._browse)
        self.cmb_class.currentTextChanged.connect(self._class_changed)

    # ------------------------------------------------------------------ folder logic
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select segments folder", str(Path.home()))
        if path:
            self.load_folder(Path(path))

    def load_folder(self, folder: Path) -> None:
        self.folder = folder
        self.lbl_folder.setText(str(folder))
        self.cmb_class.blockSignals(True)
        self.cmb_class.clear()
        for sub in sorted(folder.iterdir()):
            if sub.is_dir():
                self.cmb_class.addItem(sub.name, sub)
        self.cmb_class.blockSignals(False)
        if self.cmb_class.count():
            self.cmb_class.setCurrentIndex(0)

    # ------------------------------------------------------------------ class logic
    def _class_changed(self, _: str) -> None:
        path = self.cmb_class.currentData()
        self.csvs = []
        if path:
            self.csvs = sorted(path.glob("*.csv"))
        self.current_index = 0
        self._load_current()

    # ------------------------------------------------------------------ load/display
    def _load_current(self) -> None:
        if not self.csvs:
            self.fig.clf()
            self.canvas.draw_idle()
            for lbl in self.lbl_peaks:
                lbl.setText("No image")
                lbl.clear()
            return
        csv = self.csvs[self.current_index]
        self._plot_csv(csv)
        self._load_peaks(csv)

    def _plot_csv(self, csv: Path) -> None:
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        try:
            df = pd.read_csv(csv)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            for col in ("accel_x", "accel_y", "accel_z"):
                if col in df.columns:
                    ax.plot(df["time"], df[col], label=col)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("m/s²")
            ax.legend(fontsize="x-small")
        self.canvas.draw_idle()

    def _load_peaks(self, csv: Path) -> None:
        for lbl in self.lbl_peaks:
            lbl.clear()
            lbl.setText("No image")
        seg_id = csv.stem.split("seg")[-1]
        peak_dir = csv.parent / "peaks" / f"seg{seg_id}"
        if peak_dir.is_dir():
            for lbl, img in zip(self.lbl_peaks, sorted(peak_dir.glob("*.png"))):
                pix = QPixmap(str(img))
                if lbl.width() and lbl.height():
                    pix = pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pix)

    # ------------------------------------------------------------------ arrow keys
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Right and self.csvs:
            self.current_index = min(self.current_index + 1, len(self.csvs) - 1)
            self._load_current()
            return
        if event.key() == Qt.Key_Left and self.csvs:
            self.current_index = max(self.current_index - 1, 0)
            self._load_current()
            return
        super().keyPressEvent(event)
