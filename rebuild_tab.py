from __future__ import annotations

from pathlib import Path
from typing import Dict

from labels import (
    ANOMALY_TYPES, LABEL_IDS,
    UNKNOWN_ID, UNKNOWN_NAME, UNKNOWN_COLOR,
)

import pandas as pd
from matplotlib.widgets import SpanSelector
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QTabWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import shutil

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ---------------------------------------------------------------------------
class RebuildTab(QWidget):
    """Load exported CSV data to adjust labels and view peak images."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.folder: Path | None = None
        self.csv_dfs: Dict[str, pd.DataFrame] = {}
        self.current_csv: str | None = None
        self.current_span: tuple[float, float] | None = None
        self.view_mode = "Raw"

        vbox = QVBoxLayout(self)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel("-")
        hl.addWidget(self.lbl_folder)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        hl.addWidget(self.btn_browse)
        hl.addStretch()
        vbox.addLayout(hl)

        self.tabs = QTabWidget()
        vbox.addWidget(self.tabs, 1)

        # --------------------------- Data tab
        w_data = QWidget()
        v_data = QVBoxLayout(w_data)
        self.cmb_csv = QComboBox()
        self.cmb_csv.currentTextChanged.connect(self._select_csv)
        v_data.addWidget(self.cmb_csv)

        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Raw", "Corrected", "Weighted"])
        self.cmb_view.currentTextChanged.connect(self._update_view)
        v_data.addWidget(self.cmb_view)

        self.fig = Figure(layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        v_data.addWidget(self.canvas, 1)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Label:"))
        self.cmb_label = QComboBox()
        self.cmb_label.addItems(list(ANOMALY_TYPES))
        ctl.addWidget(self.cmb_label)
        self.btn_add = QPushButton("Add")
        self.btn_add.clicked.connect(self._add_label)
        ctl.addWidget(self.btn_add)
        self.btn_del = QPushButton("Delete")
        self.btn_del.clicked.connect(self._del_label)
        ctl.addWidget(self.btn_del)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save_csv)
        ctl.addWidget(self.btn_save)
        self.btn_export = QPushButton("Export…")
        self.btn_export.clicked.connect(self._export)
        ctl.addWidget(self.btn_export)
        ctl.addStretch()
        v_data.addLayout(ctl)

        self.tabs.addTab(w_data, "Data")

        # --------------------------- Peak image tab
        w_peak = QWidget()
        v_peak = QVBoxLayout(w_peak)
        self.cmb_peak = QComboBox()
        self.cmb_peak.currentTextChanged.connect(self._select_peak)
        v_peak.addWidget(self.cmb_peak)
        self.lbl_peak = QLabel("No image")
        self.lbl_peak.setAlignment(Qt.AlignCenter)
        v_peak.addWidget(self.lbl_peak, 1)
        self.tabs.addTab(w_peak, "Peaks")

        self.peak_imgs: list[Path] = []
        self.peak_index = 0
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ browsing
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select export folder", str(Path.home()))
        if path:
            self.load_folder(Path(path))

    def load_folder(self, folder: Path) -> None:
        self.folder = folder
        self.lbl_folder.setText(str(folder))
        self.csv_dfs.clear()
        self.cmb_csv.clear()
        for csv in sorted(folder.glob("*.csv")):
            if csv.name.endswith("_track.csv"):
                continue
            self.cmb_csv.addItem(csv.name, csv)
        # peaks
        self.cmb_peak.clear()
        peak_dir = folder / "peaks"
        if peak_dir.is_dir():
            for sub in sorted(peak_dir.iterdir()):
                if sub.is_dir():
                    self.cmb_peak.addItem(sub.name, sub)

    # ------------------------------------------------------------------ CSV logic
    def _select_csv(self, name: str) -> None:
        path = self.cmb_csv.currentData()
        if not path:
            return
        df = pd.read_csv(path)
        self.csv_dfs[name] = df
        self.current_csv = name
        self._draw(df)

    def _update_view(self, mode: str) -> None:
        self.view_mode = mode
        if self.current_csv:
            df = self.csv_dfs[self.current_csv]
            self._draw(df)

    def _draw(self, df: pd.DataFrame) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if self.view_mode == "Raw":
            ax.plot(df["time"], df["accel_x"], label="accel_x")
            ax.plot(df["time"], df["accel_y"], label="accel_y")
            ax.plot(df["time"], df["accel_z"], label="accel_z")
        elif self.view_mode == "Corrected" and "accel_corr_x" in df.columns:
            ax.plot(df["time"], df["accel_corr_x"], label="accel_corr_x")
            ax.plot(df["time"], df["accel_corr_y"], label="accel_corr_y")
            ax.plot(df["time"], df["accel_corr_z"], label="accel_corr_z")
        elif self.view_mode == "Weighted" and "awx" in df.columns:
            ax.plot(df["time"], df["awx"], label="awx")
            ax.plot(df["time"], df["awy"], label="awy")
            ax.plot(df["time"], df["awz"], label="awz")
            if "awv" in df.columns:
                ax.plot(df["time"], df["awv"], label="awv")
        else:
            ax.plot(df["time"], df["accel_x"], label="accel_x")
            ax.plot(df["time"], df["accel_y"], label="accel_y")
            ax.plot(df["time"], df["accel_z"], label="accel_z")
        self._restore_labels(ax, df)
        self.span = SpanSelector(ax, self._span, "horizontal", useblit=True,
                                 props=dict(alpha=.3, facecolor="#ff8888"))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("m/s²")
        ax.legend(fontsize="x-small")
        self.canvas.draw_idle()

    def _restore_labels(self, ax, df: pd.DataFrame) -> None:
        last = UNKNOWN_NAME
        start = None
        prev = None
        for t, lbl in zip(df["time"], df["label_name"]):
            if lbl != last:
                if last != UNKNOWN_NAME and start is not None:
                    color = ANOMALY_TYPES[last]["color"]
                    ax.axvspan(start, prev, color=color, alpha=0.2)
                start = t
                last = lbl
            prev = t
        if last != UNKNOWN_NAME and start is not None and prev is not None:
            color = ANOMALY_TYPES[last]["color"]
            ax.axvspan(start, prev, color=color, alpha=0.2)

    def _span(self, xmin: float, xmax: float) -> None:
        self.current_span = (xmin, xmax)

    def _add_label(self) -> None:
        if not self.current_csv or not self.current_span:
            return
        df = self.csv_dfs[self.current_csv]
        xmin, xmax = self.current_span
        if xmax <= xmin:
            return
        lname = self.cmb_label.currentText()
        lid = LABEL_IDS[lname]
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        df.loc[mask, ["label_id", "label_name"]] = [lid, lname]
        self._draw(df)

    def _del_label(self) -> None:
        if not self.current_csv or not self.current_span:
            return
        df = self.csv_dfs[self.current_csv]
        xmin, xmax = self.current_span
        if xmax <= xmin:
            return
        mask = (df["time"] >= xmin) & (df["time"] <= xmax)
        df.loc[mask, ["label_id", "label_name"]] = [UNKNOWN_ID, UNKNOWN_NAME]
        self._draw(df)

    def _save_csv(self) -> None:
        if not self.current_csv:
            return
        path = self.cmb_csv.currentData()
        df = self.csv_dfs[self.current_csv]
        df.to_csv(path, index=False)

    def _export(self) -> None:
        if not self.folder:
            return
        dest = QFileDialog.getExistingDirectory(
            self, "Select target folder", str(self.folder.parent)
        )
        if not dest:
            return
        dst = Path(dest)
        dst.mkdir(parents=True, exist_ok=True)
        for item in self.folder.iterdir():
            if item.is_dir():
                shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst / item.name)

    # ------------------------------------------------------------------ peaks
    def _select_peak(self, name: str) -> None:
        path = self.cmb_peak.currentData()
        if not path:
            self.peak_imgs = []
            self.lbl_peak.setText("No image")
            self.lbl_peak.setPixmap(QPixmap())
            return
        self.peak_imgs = sorted(path.glob("*.png"))
        self.peak_index = 0
        self._show_peak_image()

    def _show_peak_image(self) -> None:
        if self.peak_imgs:
            img = self.peak_imgs[self.peak_index]
            pix = QPixmap(str(img))
            self.lbl_peak.setPixmap(pix.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_peak.setText("")
        else:
            self.lbl_peak.setText("No image")
            self.lbl_peak.setPixmap(QPixmap())

    def keyPressEvent(self, event) -> None:
        if self.tabs.currentIndex() == 1 and self.peak_imgs:
            if event.key() == Qt.Key_Right:
                self.peak_index = (self.peak_index + 1) % len(self.peak_imgs)
                self._show_peak_image()
                return
            if event.key() == Qt.Key_Left:
                self.peak_index = (self.peak_index - 1) % len(self.peak_imgs)
                self._show_peak_image()
                return
        super().keyPressEvent(event)
