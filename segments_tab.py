from __future__ import annotations
import math, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum   import Enum

import pandas as pd
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui  import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QGridLayout, QSizePolicy, QSplitter
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:                                   # Matplotlib < 3.8
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from labels import ANOMALY_TYPES

# --------------------------------------------------------------------------
#    ──►  Index = genau drei Ziffern am Dateiende         (…_010.png)
IDX_PAT  = re.compile(r"_(\d{3})\.png$", re.I)

#    ──►  erweiterte View‑Erkennung:
#         zuerst komplette Kamera‑Bezeichner,
#         danach die generischen Richtungen
VIEW_PAT = re.compile(
    r"(?:"
    r"image_raw_front|image_raw_left|image_raw_right|"  # ROS Cam
    r"zed_left|zed_right|zed_rear|"                    # ZED Cam
    r"front|rear|left|right|veh|vehicle|road"          # generisch
    r")",
    re.I,
)

# abgebildete Reihenfolge im Grid
VIEWS = [
    "image_raw_front", "image_raw_left", "image_raw_right",
    "zed_left", "zed_right", "zed_rear",
    "front", "left", "right", "rear", "veh", "road"
]

# optionale Aliase (z. B. vehicle → veh)
ALIAS = dict(vehicle="veh")

# --------------------------------------------------------------------------
class ViewMode(Enum):
    RAW, CORRECTED, VEHICLE, WEIGHTED = range(4)

    @classmethod
    def titles(cls) -> List[str]:
        return ["Raw", "Corrected", "Vehicle", "Weighted"]

# --------------------------------------------------------------------------
class SegmentsTab(QWidget):
    """Signal‑Plot + synchronisierte Mehrfach‑Bilder (1 … 12 Sichten)."""

    # ------------------------------------------------------------ init
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # --- Daten
        self.root_folder: Optional[Path] = None
        self.label_folders: Dict[str, List[Path]] = {}
        self.current_label, self.current_csv = "", ""
        self.seg_idx, self.frame_pos = 0, 0
        self.view_mode: ViewMode = ViewMode.RAW
        self.current_df: Optional[pd.DataFrame] = None
        self.img_paths: Dict[Tuple[str, str], Path] = {}      # (view, idx) → Path
        self.frame_indices: List[str] = []
        self.pix_cache: Dict[Tuple[str, str], QPixmap] = {}

        # --- UI
        self._build_ui()
        self._connect()
        self.setFocusPolicy(Qt.StrongFocus)

    # ======================================================================
    # UI‑Aufbau
    # ======================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ---------- Zeile: Ordner & CSV
        row = QHBoxLayout()
        row.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel("-")
        row.addWidget(self.lbl_folder, 1)
        row.addWidget(QLabel("CSV:"))
        self.lbl_csv = QLabel("-")
        row.addWidget(self.lbl_csv, 1)
        self.btn_browse = QPushButton("Browse…")
        row.addWidget(self.btn_browse)
        root.addLayout(row)

        # ---------- Zeile: Label‑Ordner + Segment‑Label
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Label folder:"))
        self.cmb_folder = QComboBox()
        row2.addWidget(self.cmb_folder)
        row2.addWidget(QLabel("Segment label:"))
        self.cmb_label = QComboBox()
        self.cmb_label.addItems(list(ANOMALY_TYPES))
        row2.addWidget(self.cmb_label)
        root.addLayout(row2)

        # ---------- Splitter: Plot oben | Bilder unten
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, 1)

        # --- Plot‑Pane
        plot_pane = QWidget()
        pv = QVBoxLayout(plot_pane); pv.setContentsMargins(0, 0, 0, 0)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("View:"))
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(ViewMode.titles())
        row3.addWidget(self.cmb_view)
        row3.addStretch()
        pv.addLayout(row3)

        self.fig = Figure(figsize=(9, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        pv.addWidget(self.canvas)
        splitter.addWidget(plot_pane)

        # --- Bilder‑Pane (Grid wird dynamisch neu gebaut)
        self.img_container = QWidget()
        self.img_layout    = QGridLayout(self.img_container)
        self.img_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(self.img_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _connect(self) -> None:
        self.btn_browse.clicked.connect(self._browse)
        self.cmb_folder.currentTextChanged.connect(self._folder_changed)
        self.cmb_view.currentTextChanged.connect(self._change_view)

    # ======================================================================
    # Ordner‑Handling
    # ======================================================================
    def _browse(self) -> None:
        p = QFileDialog.getExistingDirectory(
            self, "Select segments folder", str(Path.home()))
        if p: 
            self.load_folder(Path(p))

    def load_folder(self, folder: Path) -> None:
        self.root_folder = folder
        self.lbl_folder.setText(str(folder))

        self.label_folders.clear()
        self.cmb_folder.blockSignals(True)
        self.cmb_folder.clear()

        for sub in sorted(folder.iterdir()):
            if sub.is_dir():
                csvs = sorted(sub.glob("*.csv"))
                if csvs:
                    self.label_folders[sub.name] = csvs
                    self.cmb_folder.addItem(sub.name)

        self.cmb_folder.blockSignals(False)
        if self.label_folders:
            self.cmb_folder.setCurrentIndex(0)
            self._folder_changed(self.cmb_folder.currentText())

    def _folder_changed(self, label: str) -> None:
        self.current_label, self.seg_idx = label, 0
        self._load_segment()

    # ======================================================================
    # Segment‑ und Frame‑Logik
    # ======================================================================
    def _load_segment(self) -> None:
        # Reset Anzeige
        self.fig.clf(); self.canvas.draw_idle()
        self._clear_images()

        csvs = self.label_folders.get(self.current_label, [])
        if not csvs: 
            return

        csv = csvs[self.seg_idx]
        self.lbl_csv.setText(csv.name)
        try:
            self.current_df = pd.read_csv(csv)
        except Exception:
            self.current_df = None
            return

        self.cmb_label.setCurrentText(self.current_label)
        self._draw_plot()

        # ----- Peaks sammeln
        self.img_paths.clear(); self.pix_cache.clear()
        seg_id = csv.stem.split("_")[-1]          # seg###
        peak_dir = csv.parent / "peaks" / seg_id
        if peak_dir.is_dir():
            for p in peak_dir.glob("*.png"):
                idx  = self._idx(p.name)
                view = self._view(p.name)
                if idx and view:
                    self.img_paths[(view, idx)] = p

        self.frame_indices = sorted({i for _, i in self.img_paths})
        self.frame_pos = 0
        self._show_frame()

    # ------------------------------------------------------------------ helper
    @staticmethod
    def _idx(name: str) -> str | None:
        """liefert z. B. '010'"""
        m = IDX_PAT.search(name)
        return m.group(1) if m else None

    @staticmethod
    def _view(name: str) -> str | None:
        """liefert eindeutigen View‑Key – inkl. Kamera‑Quelle"""
        m = VIEW_PAT.search(name)
        if not m:
            return None
        v_raw = m.group(0).lower()
        v = ALIAS.get(v_raw, v_raw)
        return v if v in VIEWS else None

    # ======================================================================
    # Plot
    # ======================================================================
    def _change_view(self, txt: str) -> None:
        try:
            self.view_mode = ViewMode[txt.upper()]
        except KeyError:
            self.view_mode = ViewMode.RAW
        self._draw_plot()

    def _draw_plot(self) -> None:
        self.fig.clf()
        if self.current_df is None:
            self.canvas.draw_idle()
            return

        cols_map = {
            ViewMode.RAW:       ("accel_x",       "accel_y",       "accel_z"),
            ViewMode.CORRECTED: ("accel_corr_x",  "accel_corr_y",  "accel_corr_z"),
            ViewMode.VEHICLE:   ("accel_veh_x",   "accel_veh_y",   "accel_veh_z"),
            ViewMode.WEIGHTED:  ("awx", "awy", "awz", "awv"),
        }
        cols = cols_map.get(self.view_mode, cols_map[ViewMode.RAW])

        ax = self.fig.add_subplot(111)
        for c in cols:
            if c in self.current_df.columns:
                ax.plot(self.current_df["time"], self.current_df[c], label=c)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("m/s²")
        ax.legend(fontsize="x-small")
        self.canvas.draw_idle()

    # ======================================================================
    # Bilder
    # ======================================================================
    def _clear_images(self):
        while self.img_layout.count():
            w = self.img_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

    def _show_frame(self) -> None:
        self._clear_images()
        if not self.frame_indices:
            return

        idx = self.frame_indices[self.frame_pos]

        # vorhandene Ansichten sammeln
        present = [(view, self.img_paths[(view, idx)])
                   for view in VIEWS if (view, idx) in self.img_paths]

        n = len(present)
        if not n:
            return

        # Grid‑Dimensionen: 1→1, 2→2, 3‑6→3, 7‑9→3, 10‑12→4 …
        cols = 3 if n <= 9 else 4
        rows = math.ceil(n / cols)

        for k, (view, path) in enumerate(present):
            r, c = divmod(k, cols)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setMinimumSize(QSize(160, 120))
            self.img_layout.addWidget(lbl, r, c)

            key = (view, idx)
            if key not in self.pix_cache:
                self.pix_cache[key] = QPixmap(str(path))
            pix = self.pix_cache[key]
            lbl.setPixmap(
                pix.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            lbl.setToolTip(
                f"{view} | Frame {self.frame_pos+1}/{len(self.frame_indices)}"
            )

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._show_frame()

    # ======================================================================
    # Tastatur‑Navigation
    # ======================================================================
    def keyPressEvent(self, ev) -> None:  # noqa: N802
        k = ev.key()
        if k == Qt.Key_Right:
            self._next_seg();  return
        if k == Qt.Key_Left:
            self._prev_seg();  return
        if k == Qt.Key_Up:
            self._next_frame(); return
        if k == Qt.Key_Down:
            self._prev_frame(); return
        super().keyPressEvent(ev)

    def _next_seg(self):
        lst = self.label_folders.get(self.current_label, [])
        if self.seg_idx + 1 < len(lst):
            self.seg_idx += 1
            self._load_segment()

    def _prev_seg(self):
        if self.seg_idx > 0:
            self.seg_idx -= 1
            self._load_segment()

    def _next_frame(self):
        if self.frame_pos + 1 < len(self.frame_indices):
            self.frame_pos += 1
            self._show_frame()

    def _prev_frame(self):
        if self.frame_pos > 0:
            self.frame_pos -= 1
            self._show_frame()
