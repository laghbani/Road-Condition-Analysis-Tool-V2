from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class StatsTab(QWidget):
    """Display class statistics from exported CSV files."""

    def __init__(self, colors: dict[str, str], unknown_name: str, parent=None) -> None:
        super().__init__(parent)
        vbox = QVBoxLayout(self)

        # ----- control bar -----
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Folder:"))
        self.lbl_path = QLabel("-")
        hl.addWidget(self.lbl_path)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        hl.addWidget(self.btn_browse)
        hl.addStretch()
        vbox.addLayout(hl)

        # ----- plot -----
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)

        self.colors = colors
        self.unknown_name = unknown_name

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select CSV Folder")
        if path:
            self.load_folder(path)

    def load_folder(self, folder: str | Path) -> None:
        folder = Path(folder)
        self.lbl_path.setText(str(folder))
        csv_files = [p for p in folder.glob("*.csv") if not p.name.endswith("_track.csv")]
        dp_counts = defaultdict(int)
        grp_counts = defaultdict(int)
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if "label_name" not in df.columns:
                continue
            labels = df["label_name"].fillna(self.unknown_name)
            for lbl, cnt in labels.value_counts().items():
                dp_counts[str(lbl)] += int(cnt)
            seg_id = (labels != labels.shift()).cumsum()
            for _, grp in labels.groupby(seg_id):
                grp_counts[str(grp.iloc[0])] += 1
        self._plot_stats(dp_counts, grp_counts)

    def _plot_stats(self, dp_counts: dict[str, int], grp_counts: dict[str, int]) -> None:
        self.fig.clear()
        if not dp_counts and not grp_counts:
            self.canvas.draw()
            return
        ax1 = self.fig.add_subplot(211)
        names = list(dp_counts)
        vals = [dp_counts[n] for n in names]
        cols = [self.colors.get(n, "#808080") for n in names]
        ax1.bar(names, vals, color=cols)
        ax1.set_title("Data Points per Class")
        ax1.tick_params(axis='x', rotation=45)

        ax2 = self.fig.add_subplot(212)
        names_g = list(grp_counts)
        vals_g = [grp_counts[n] for n in names_g]
        cols_g = [self.colors.get(n, "#808080") for n in names_g]
        ax2.bar(names_g, vals_g, color=cols_g)
        ax2.set_title("Groups per Class")
        ax2.tick_params(axis='x', rotation=45)

        self.fig.tight_layout()
        self.canvas.draw()
