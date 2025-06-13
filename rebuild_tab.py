"""
rebuild_tab.py – CSV label editor & peak viewer (English GUI)

Public API (class / method / attribute names) unchanged → main_gui keeps working.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import shutil

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Project‑specific label mapping
# ---------------------------------------------------------------------------
from labels import (  # noqa: E402  – local import after Qt setup
    ANOMALY_TYPES,
    LABEL_IDS,
    UNKNOWN_ID,
    UNKNOWN_NAME,
)

# -----------------------------------------------------------------------------
# Helper Enum
# -----------------------------------------------------------------------------


class ViewMode(Enum):
    RAW = auto()
    CORRECTED = auto()
    WEIGHTED = auto()

    @classmethod
    def titles(cls) -> List[str]:
        return [m.name.capitalize() for m in cls]


# -----------------------------------------------------------------------------
# Small dialog to change a label
# -----------------------------------------------------------------------------


class LabelEditDialog(QDialog):
    """Dialog that lets the user change a label."""

    def __init__(self, current: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Label")

        layout = QHBoxLayout(self)

        layout.addWidget(QLabel(current))

        self.cmb: QComboBox = QComboBox()
        self.cmb.addItems(list(ANOMALY_TYPES))
        self.cmb.setCurrentText(current)
        layout.addWidget(self.cmb)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # keep method name so external code still compiles
    def result(self) -> str:  # noqa: D401
        """Return the selected label name."""
        return self.cmb.currentText()


# -----------------------------------------------------------------------------
# Main widget
# -----------------------------------------------------------------------------


class RebuildTab(QWidget):
    """Load exported CSVs, allow relabeling via span selection and show peaks."""

    # ------------------------------------------------------------------ init
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # ---------- Data containers ----------------------------------------------
        self.folder: Optional[Path] = None
        self.csv_dfs: Dict[str, pd.DataFrame] = {}
        self.csv_paths: Dict[str, Path] = {}
        self.current_csv: Optional[str] = None
        self.current_span: Optional[Tuple[float, float]] = None
        self.view_mode: ViewMode = ViewMode.RAW
        self.view_csv: Optional[str] = None

        # Matplotlib axis → csv mapping
        self.ax_csv: Dict[object, str] = {}
        self.span_selectors: Dict[str, SpanSelector] = {}

        # ---------- Peaks --------------------------------------------------------
        self.peak_groups: List[List[Path]] = []
        self.peak_index: int = 0

        # ---------- UI setup -----------------------------------------------------
        self._build_ui()
        self._connect_signals()
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------ UI helpers
    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)

        # ---- folder line -------------------------------------------------------
        hl_path = QHBoxLayout()
        hl_path.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel("-")
        hl_path.addWidget(self.lbl_folder, 1)
        self.btn_browse = QPushButton("Browse…")
        hl_path.addWidget(self.btn_browse)
        vbox.addLayout(hl_path)

        # ---- tab widget --------------------------------------------------------
        self.tabs = QTabWidget()
        vbox.addWidget(self.tabs, 1)

        # ===================== Tab 1 : Data =====================================
        w_data = QWidget()
        v_data = QVBoxLayout(w_data)

        self.cmb_csv = QComboBox()
        v_data.addWidget(self.cmb_csv)

        self.cmb_view = QComboBox()
        self.cmb_view.addItems(ViewMode.titles())
        v_data.addWidget(self.cmb_view)

        opt = QHBoxLayout()
        self.chk_names = QCheckBox("Show names")
        self.chk_legend = QCheckBox("Show legend")
        self.chk_xlabel = QCheckBox("Show time axis")
        opt.addWidget(self.chk_names)
        opt.addWidget(self.chk_legend)
        opt.addWidget(self.chk_xlabel)
        opt.addStretch()
        v_data.addLayout(opt)

        self.fig = Figure(figsize=(9, 6), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(400)
        v_data.addWidget(self.canvas, 1)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Label:"))
        self.cmb_label = QComboBox()
        self.cmb_label.addItems(list(ANOMALY_TYPES))
        ctl.addWidget(self.cmb_label)

        self.btn_add = QPushButton("Add")
        ctl.addWidget(self.btn_add)
        self.btn_del = QPushButton("Delete")
        ctl.addWidget(self.btn_del)
        self.btn_edit = QPushButton("Edit…")
        ctl.addWidget(self.btn_edit)
        self.btn_save = QPushButton("Save")
        ctl.addWidget(self.btn_save)
        self.btn_export = QPushButton("Export…")
        ctl.addWidget(self.btn_export)
        ctl.addStretch()
        v_data.addLayout(ctl)

        self.tabs.addTab(w_data, "Data")

        # ===================== Tab 2 : Peaks ====================================
        w_peak = QWidget()
        v_peak = QVBoxLayout(w_peak)

        self.cmb_peak = QComboBox()
        v_peak.addWidget(self.cmb_peak)

        grid = QGridLayout()
        self.lbl_peaks: List[QLabel] = []
        for i in range(6):
            lbl = QLabel("No image")
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, i // 3, i % 3)
            self.lbl_peaks.append(lbl)
        v_peak.addLayout(grid)

        self.tabs.addTab(w_peak, "Peaks")

    def _connect_signals(self) -> None:
        self.btn_browse.clicked.connect(self._browse)

        self.cmb_csv.currentTextChanged.connect(self._select_csv)
        self.cmb_view.currentTextChanged.connect(self._update_view)

        self.btn_add.clicked.connect(self._add_label)
        self.btn_del.clicked.connect(self._del_label)
        self.btn_edit.clicked.connect(self._edit_label)
        self.btn_save.clicked.connect(self._save_csv)
        self.btn_export.clicked.connect(self._export)

        self.canvas.mpl_connect("button_press_event", self._mouse_press)

        self.cmb_peak.currentTextChanged.connect(self._select_peak)

    # ------------------------------------------------------------------ folder logic
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select export folder", str(Path.home())
        )
        if path:
            self.load_folder(Path(path))

    def load_folder(self, folder: Path) -> None:
        """Read all CSV + peak data from folder."""
        self.folder = folder
        self.lbl_folder.setText(str(folder))

        self.csv_dfs.clear()
        self.csv_paths.clear()
        self.cmb_csv.blockSignals(True)
        self.cmb_csv.clear()

        for csv in sorted(folder.glob("*.csv")):
            if csv.name.endswith("_track.csv"):
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as err:  # noqa: BLE001
                QMessageBox.critical(
                    self,
                    "CSV error",
                    f"Could not read '{csv.name}':\n{err}",
                )
                continue
            self.csv_dfs[csv.name] = df
            self.csv_paths[csv.name] = csv

        self.cmb_csv.addItem("(All)", None)
        for name in sorted(self.csv_dfs):
            self.cmb_csv.addItem(name, self.csv_paths[name])
        self.cmb_csv.blockSignals(False)

        # peaks
        self.cmb_peak.blockSignals(True)
        self.cmb_peak.clear()
        peak_dir = folder / "peaks"
        if peak_dir.is_dir():
            for sub in sorted(peak_dir.iterdir()):
                if sub.is_dir():
                    self.cmb_peak.addItem(sub.name, sub)
        self.cmb_peak.blockSignals(False)

        self.current_csv = None
        self.current_span = None
        self._draw()

    # ------------------------------------------------------------------ CSV view
    def _select_csv(self, name: str) -> None:
        self.view_csv = None if name in {"(All)", ""} else name
        self.current_csv = self.view_csv
        self._draw()

    def _update_view(self, text: str) -> None:
        try:
            self.view_mode = ViewMode[text.upper()]
        except KeyError:
            self.view_mode = ViewMode.RAW
        self._draw()

    # ------------------------------------------------------------------ plotting
    def _draw(self) -> None:
        self.fig.clf()
        self.ax_csv.clear()

        for sel in self.span_selectors.values():
            sel.disconnect_events()
        self.span_selectors.clear()

        dfs = (
            {self.view_csv: self.csv_dfs[self.view_csv]}
            if self.view_csv
            else self.csv_dfs
        )
        if not dfs:
            self.canvas.draw_idle()
            return

        gs = self.fig.add_gridspec(len(dfs), 1, hspace=0.30)
        for i, (name, df) in enumerate(dfs.items()):
            ax = self.fig.add_subplot(gs[i])
            self._plot_df(ax, df)
            if self.chk_names.isChecked():
                ax.set_title(name, fontsize="small")
            self.ax_csv[ax] = name
            self.span_selectors[name] = SpanSelector(
                ax,
                lambda x0, x1, n=name: self._span_selected(n, x0, x1),
                "horizontal",
                useblit=True,
                props=dict(alpha=0.30, facecolor="#ff8888"),
            )

        self.canvas.draw_idle()

    def _plot_df(self, ax, df: pd.DataFrame) -> None:
        columns_by_mode = {
            ViewMode.RAW: ("accel_x", "accel_y", "accel_z"),
            ViewMode.CORRECTED: ("accel_corr_x", "accel_corr_y", "accel_corr_z"),
            ViewMode.WEIGHTED: ("awx", "awy", "awz", "awv"),
        }
        cols = columns_by_mode.get(self.view_mode, columns_by_mode[ViewMode.RAW])

        for col in cols:
            if col in df.columns:
                ax.plot(df["time"], df[col], label=col)

        self._restore_labels(ax, df)

        if self.chk_xlabel.isChecked():
            ax.set_xlabel("time [s]")
        ax.set_ylabel("m/s²")
        if self.chk_legend.isChecked():
            ax.legend(fontsize="x-small")

    def _restore_labels(self, ax, df: pd.DataFrame) -> None:
        last, start, prev = UNKNOWN_NAME, None, None
        for t, lbl in zip(df["time"], df["label_name"]):
            if lbl != last:
                self._draw_span(ax, last, start, prev)
                start, last = t, lbl
            prev = t
        self._draw_span(ax, last, start, prev)

    @staticmethod
    def _draw_span(ax, lbl: str, start: Optional[float], end: Optional[float]) -> None:
        if (
            lbl != UNKNOWN_NAME
            and start is not None
            and end is not None
            and lbl in ANOMALY_TYPES
        ):
            ax.axvspan(start, end, color=ANOMALY_TYPES[lbl]["color"], alpha=0.2)

    # ------------------------------------------------------------------ span logic
    def _span_selected(self, csv: str, xmin: float, xmax: float) -> None:
        self.current_csv, self.current_span = csv, (xmin, xmax)

    def _selected_dfs(self) -> Iterable[pd.DataFrame]:
        """
        1) If “(All)” is chosen (self.view_csv is None) → apply to ALL files.
        2) Else only to the chosen CSV.
        """
        if self.view_csv is None:
            yield from self.csv_dfs.values()
        else:
            yield self.csv_dfs[self.view_csv]

    def _apply_range(
        self,
        start: float,
        end: float,
        lid: int,
        lname: str,
    ) -> None:
        if end <= start:
            return
        eps = 1e-3
        for df in self._selected_dfs():
            mask = (df["time"] >= start - eps) & (df["time"] <= end + eps)
            df.loc[mask, ["label_id", "label_name"]] = [lid, lname]

    # ------------------------------------------------------------------ label buttons
    def _add_label(self) -> None:
        if not self.current_span:
            return
        lid = LABEL_IDS[self.cmb_label.currentText()]
        self._apply_range(*self.current_span, lid, self.cmb_label.currentText())
        self._draw()

    def _del_label(self) -> None:
        if self.current_span:
            self._apply_range(*self.current_span, UNKNOWN_ID, UNKNOWN_NAME)
            self._draw()

    def _edit_label(self) -> None:
        """Button ‘Edit…’: opens dialog for chosen range (if any)."""
        if not self.current_span:
            return
        xmin, xmax = self.current_span
        dlg = LabelEditDialog(self.cmb_label.currentText(), self)
        if dlg.exec() == QDialog.Accepted:
            lname = dlg.result()
            lid = LABEL_IDS[lname]
            self._apply_range(xmin, xmax, lid, lname)
            self._draw()

    # ------------------------------------------------------------------ file buttons
    def _save_csv(self) -> None:
        if not self.view_csv:
            return
        path = self.csv_paths.get(self.view_csv)
        if not path:
            return
        try:
            self.csv_dfs[self.view_csv].to_csv(path, index=False)
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Save error",
                f"Could not save '{path.name}':\n{err}",
            )

    def _export(self) -> None:
        if not self.folder:
            return
        dest = QFileDialog.getExistingDirectory(
            self,
            "Select target folder",
            str(self.folder.parent),
        )
        if not dest:
            return
        dst = Path(dest)
        for item in self.folder.iterdir():
            target = dst / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
            except Exception as err:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Export warning",
                    f"Could not copy '{item.name}':\n{err}",
                )

    # ------------------------------------------------------------------ peaks
    def _select_peak(self, _: str) -> None:
        path = self.cmb_peak.currentData()
        self.peak_groups.clear()
        if path:
            groups: Dict[str, List[Path]] = {}
            for img in sorted(path.glob("*.png")):
                groups.setdefault(img.stem.split("_")[-1], []).append(img)
            self.peak_groups = [groups[k] for k in sorted(groups)]
        self.peak_index = 0
        self._show_peak_images()

    def _show_peak_images(self) -> None:
        for lbl in self.lbl_peaks:
            lbl.clear()
            lbl.setText("No image")
        if not self.peak_groups:
            return
        imgs = self.peak_groups[self.peak_index]
        for lbl, img in zip(self.lbl_peaks, imgs):
            pix = QPixmap(str(img))
            if lbl.width() and lbl.height():
                pix = pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(pix)

    def keyPressEvent(self, event) -> None:  # noqa: N802
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

    # ------------------------------------------------------------------ mouse handling
    def _mouse_press(self, event) -> None:
        """
        • double‑left‑click  → change label via dialog
        • double‑right‑click → set to UNKNOWN
        Single clicks are ignored.
        """
        if event.inaxes is None or event.xdata is None:
            return
        if not event.dblclick:
            return  # ignore single clicks

        csv = self.ax_csv.get(event.inaxes)
        if not csv:
            return
        df = self.csv_dfs[csv]
        times = df["time"].to_numpy()
        idx = int(np.clip(np.searchsorted(times, event.xdata), 0, len(df) - 1))
        lbl = df.loc[idx, "label_name"]

        # contiguous region
        i0, i1 = idx, idx
        while i0 > 0 and df.loc[i0 - 1, "label_name"] == lbl:
            i0 -= 1
        while i1 + 1 < len(df) and df.loc[i1 + 1, "label_name"] == lbl:
            i1 += 1
        start, end = df.loc[i0, "time"], df.loc[i1, "time"]

        if event.button == 1:  # left double‑click → edit
            dlg = LabelEditDialog(lbl, self)
            if dlg.exec() == QDialog.Accepted:
                new_lbl = dlg.result()
                self._apply_range(start, end, LABEL_IDS[new_lbl], new_lbl)
                self._draw()

        elif event.button == 3:  # right double‑click → delete (=unknown)
            self._apply_range(start, end, UNKNOWN_ID, UNKNOWN_NAME)
            self._draw()
