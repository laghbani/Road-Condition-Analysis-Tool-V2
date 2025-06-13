from __future__ import annotations

from pathlib import Path
from typing import Dict

from labels import (
    ANOMALY_TYPES, LABEL_IDS,
    UNKNOWN_ID, UNKNOWN_NAME, UNKNOWN_COLOR,
)

import pandas as pd
import numpy as np
from matplotlib.widgets import SpanSelector
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QTabWidget, QDialog, QDialogButtonBox,
    QGridLayout, QCheckBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import shutil

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ---------------------------------------------------------------------------
class LabelEditDialog(QDialog):
    """Dialog to change label name."""

    def __init__(self, current: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Label")
        layout = QHBoxLayout(self)
        layout.addWidget(QLabel(current))
        self.cmb = QComboBox()
        self.cmb.addItems(list(ANOMALY_TYPES))
        layout.addWidget(self.cmb)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def result(self) -> str:
        return self.cmb.currentText()

# ---------------------------------------------------------------------------
class RebuildTab(QWidget):
    """Load exported CSV data to adjust labels and view peak images."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.folder: Path | None = None
        self.csv_dfs: Dict[str, pd.DataFrame] = {}
        self.csv_paths: Dict[str, Path] = {}
        self.current_csv: str | None = None
        self.current_span: tuple[float, float] | None = None
        self.view_mode = "Raw"
        self.view_csv: str | None = None

        self.ax_csv: dict[object, str] = {}
        self.span_selectors: dict[str, SpanSelector] = {}

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

        opt = QHBoxLayout()
        self.chk_names = QCheckBox("Show names")
        self.chk_legend = QCheckBox("Show legend")
        opt.addWidget(self.chk_names)
        opt.addWidget(self.chk_legend)
        opt.addStretch()
        v_data.addLayout(opt)

        self.fig = Figure(figsize=(9, 6), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(400)
        self.canvas.mpl_connect("button_press_event", self._mouse_press)
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
        self.btn_edit = QPushButton("Edit…")
        self.btn_edit.clicked.connect(self._edit_label)
        ctl.addWidget(self.btn_edit)
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

        grid = QGridLayout()
        self.lbl_peaks = []
        for i in range(6):
            lbl = QLabel("No image")
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, i // 3, i % 3)
            self.lbl_peaks.append(lbl)
        v_peak.addLayout(grid)

        self.tabs.addTab(w_peak, "Peaks")

        self.peak_groups: list[list[Path]] = []
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
        self.csv_paths.clear()
        self.cmb_csv.clear()
        items = []
        for csv in sorted(folder.glob("*.csv")):
            if csv.name.endswith("_track.csv"):
                continue
            df = pd.read_csv(csv)
            self.csv_dfs[csv.name] = df
            self.csv_paths[csv.name] = csv
            items.append(csv.name)
        self.cmb_csv.addItem("(All)", None)
        for it in items:
            self.cmb_csv.addItem(it, self.csv_paths[it])
        self.current_csv = None
        self._draw()
        # peaks
        self.cmb_peak.clear()
        peak_dir = folder / "peaks"
        if peak_dir.is_dir():
            for sub in sorted(peak_dir.iterdir()):
                if sub.is_dir():
                    self.cmb_peak.addItem(sub.name, sub)

    # ------------------------------------------------------------------ CSV logic
    def _select_csv(self, name: str) -> None:
        if name == "(All)" or not name:
            self.view_csv = None
            self.current_csv = None
        else:
            self.view_csv = name
            self.current_csv = name
        self._draw()

    def _update_view(self, mode: str) -> None:
        self.view_mode = mode
        self._draw()

    def _draw(self) -> None:
        self.fig.clear()
        self.ax_csv.clear()
        for sel in self.span_selectors.values():
            sel.disconnect_events()
        self.span_selectors.clear()

        if self.view_csv:
            dfs = {self.view_csv: self.csv_dfs[self.view_csv]}
        else:
            dfs = self.csv_dfs

        if not dfs:
            self.canvas.draw_idle()
            return

        gs = self.fig.add_gridspec(len(dfs), 1)
        show_names = self.chk_names.isChecked()
        for i, (name, df) in enumerate(dfs.items()):
            ax = self.fig.add_subplot(gs[i])
            self._plot_df(ax, df)
            if show_names:
                ax.set_title(name)
            self.ax_csv[ax] = name
            self.span_selectors[name] = SpanSelector(
                ax,
                lambda x0, x1, t=name: self._span_selected(t, x0, x1),
                "horizontal",
                useblit=True,
                props=dict(alpha=.3, facecolor="#ff8888"),
            )
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_df(self, ax, df: pd.DataFrame) -> None:
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
        ax.set_xlabel("time [s]")
        ax.set_ylabel("m/s²")
        if self.chk_legend.isChecked():
            ax.legend(fontsize="x-small")

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

    def _span_selected(self, csv: str, xmin: float, xmax: float) -> None:
        self.current_csv = csv
        self.current_span = (xmin, xmax)

    def _selected_dfs(self):
        if self.view_csv is None:
            return self.csv_dfs.values()
        if self.current_csv:
            return [self.csv_dfs[self.current_csv]]
        return [self.csv_dfs[self.view_csv]]

    def _apply_range(self, start: float, end: float, lid: int, lname: str,
                     old_lid: int | None = None) -> None:
        eps = 1e-3
        for df in self._selected_dfs():
            mask = (df["time"] >= start - eps) & (df["time"] <= end + eps)
            if old_lid is not None:
                mask &= df["label_id"] == old_lid
            df.loc[mask, ["label_id", "label_name"]] = [lid, lname]

    def _add_label(self) -> None:
        if not self.current_span:
            return
        xmin, xmax = self.current_span
        if xmax <= xmin:
            return
        lname = self.cmb_label.currentText()
        lid = LABEL_IDS[lname]
        self._apply_range(xmin, xmax, lid, lname)
        self._draw()

    def _del_label(self) -> None:
        if not self.current_span:
            return
        xmin, xmax = self.current_span
        if xmax <= xmin:
            return
        self._apply_range(xmin, xmax, UNKNOWN_ID, UNKNOWN_NAME)
        self._draw()

    def _edit_label(self) -> None:
        if not self.current_span:
            return
        xmin, xmax = self.current_span
        for df in self._selected_dfs():
            mask = (df["time"] >= xmin) & (df["time"] <= xmax)
            if not mask.any():
                continue
            current = df.loc[mask, "label_name"].mode().iat[0]
            dlg = LabelEditDialog(current, self)
            if dlg.exec() != QDialog.Accepted:
                return
            lname = dlg.result()
            lid = LABEL_IDS[lname]
            self._apply_range(xmin, xmax, lid, lname)
            self._draw()
            break

    def _save_csv(self) -> None:
        if not self.current_csv:
            return
        path = self.csv_paths.get(self.current_csv)
        if path:
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
            self.peak_groups = []
            for lbl in self.lbl_peaks:
                lbl.setText("No image")
                lbl.setPixmap(QPixmap())
            return
        groups: Dict[str, list[Path]] = {}
        for img in sorted(path.glob("*.png")):
            key = img.stem.split("_")[-1]
            groups.setdefault(key, []).append(img)
        self.peak_groups = [groups[k] for k in sorted(groups.keys())]
        self.peak_index = 0
        self._show_peak_images()

    def _show_peak_images(self) -> None:
        for lbl in self.lbl_peaks:
            lbl.setPixmap(QPixmap())
            lbl.setText("No image")
        if not self.peak_groups:
            return
        imgs = self.peak_groups[self.peak_index]
        for lbl, img in zip(self.lbl_peaks, imgs):
            pix = QPixmap(str(img))
            if lbl.width() and lbl.height():
                pix = pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(pix)
            lbl.setText("")

    def keyPressEvent(self, event) -> None:
        if self.tabs.currentIndex() == 1 and self.peak_groups:
            if event.key() == Qt.Key_Right:
                self.peak_index = min(self.peak_index + 1, len(self.peak_groups) - 1)
                self._show_peak_images()
                return
            if event.key() == Qt.Key_Left:
                self.peak_index = max(self.peak_index - 1, 0)
                self._show_peak_images()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ mouse
    def _mouse_press(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        csv = self.ax_csv.get(event.inaxes)
        if not csv:
            return
        self.current_csv = csv
        df = self.csv_dfs[csv]
        times = df["time"].to_numpy()
        idx = int(np.clip(np.searchsorted(times, event.xdata), 0, len(df) - 1))
        lbl = df.loc[idx, "label_name"]
        # determine contiguous region around index
        i0 = idx
        while i0 > 0 and df.loc[i0 - 1, "label_name"] == lbl:
            i0 -= 1
        i1 = idx
        while i1 + 1 < len(df) and df.loc[i1 + 1, "label_name"] == lbl:
            i1 += 1
        start = df.loc[i0, "time"]
        end = df.loc[i1, "time"]
        old_id = df.loc[i0, "label_id"]

        if event.button == 1:
            dlg = LabelEditDialog(lbl, self)
            if dlg.exec() != QDialog.Accepted:
                return
            lname = dlg.result()
            lid = LABEL_IDS[lname]
            self._apply_range(start, end, lid, lname, old_id)
            self._draw()
        elif event.button == 3 and event.dblclick:
            self._apply_range(start, end, UNKNOWN_ID, UNKNOWN_NAME, old_id)
            self._draw()
        
