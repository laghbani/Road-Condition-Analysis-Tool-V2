from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


class StatsTab(QWidget):
    """Visualize data point totals, labeled counts and groups for each class."""

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
        self.btn_pdf = QPushButton("Save PDF…")
        self.btn_pdf.clicked.connect(self._save_pdf)
        hl.addWidget(self.btn_pdf)
        hl.addStretch()
        vbox.addLayout(hl)

        hl2 = QHBoxLayout()
        self.lbl_totals = QLabel("")
        hl2.addWidget(self.lbl_totals)
        hl2.addStretch()
        vbox.addLayout(hl2)

        # ----- plot -----
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)

        self.colors = colors
        self.unknown_name = unknown_name

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select CSV Folder")
        if path:
            self.load_folder(path)

    def _save_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot as PDF", "", "PDF Files (*.pdf)")
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self.fig.savefig(path)

    def load_folder(self, folder: str | Path) -> None:
        folder = Path(folder)
        self.lbl_path.setText(str(folder))
        csv_files = list(folder.rglob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.endswith("_track.csv")]

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

            # --- data point counts ---
            for lbl, cnt in labels.value_counts().items():
                dp_counts[str(lbl)] += int(cnt)

            # --- group counts ---
            seg_id = (labels != labels.shift()).cumsum()
            for _, grp in labels.groupby(seg_id):
                grp_counts[str(grp.iloc[0])] += 1

        total = sum(dp_counts.values())
        unknown = dp_counts.get(self.unknown_name, 0)
        self.lbl_totals.setText(f"Total: {total} | Unknown: {unknown}")

        self._plot_stats(dp_counts, grp_counts)

    def _plot_stats(
        self,
        dp_counts: dict[str, int],
        grp_counts: dict[str, int],
    ) -> None:
        self.fig.clear()
        if not dp_counts and not grp_counts:
            self.canvas.draw()
            return

        labels = [lbl for lbl in sorted(set(dp_counts) | set(grp_counts)) if lbl != self.unknown_name]
        dp_vals = [dp_counts.get(lbl, 0) for lbl in labels]
        grp_vals = [grp_counts.get(lbl, 0) for lbl in labels]
        cols = [self.colors.get(lbl, "#808080") for lbl in labels]

        # layout: pie chart on the left spanning both rows and
        # bar charts stacked on the right
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
        ax_pie = self.fig.add_subplot(gs[:, 0])
        ax_dp = self.fig.add_subplot(gs[0, 1])
        ax_grp = self.fig.add_subplot(gs[1, 1], sharex=ax_dp)

        x = range(len(labels))
        width = 0.6

        bar_dp = ax_dp.bar(x, dp_vals, color=cols)
        ax_dp.set_xticks(x)
        # x labels only on the lower plot
        ax_dp.set_xticklabels([])
        ax_dp.set_ylabel("Data Points")
        ax_dp.set_title("Data Points per Class")
        for rect, val in zip(bar_dp, dp_vals):
            ax_dp.text(rect.get_x() + rect.get_width()/2, rect.get_height(), str(val), ha="center", va="bottom", fontsize=8)

        bar_grp = ax_grp.bar(x, grp_vals, color="gray")
        ax_grp.set_xticks(x)
        ax_grp.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax_grp.set_ylabel("Groups")
        ax_grp.set_title("Groups per Class")
        for rect, val in zip(bar_grp, grp_vals):
            ax_grp.text(rect.get_x() + rect.get_width()/2, rect.get_height(), str(val), ha="center", va="bottom", fontsize=8)

        if sum(dp_vals) > 0:
            ax_pie.pie(
                dp_vals,
                labels=None,  # show only percentages
                colors=cols,
                autopct="%1.1f%%",
                startangle=140,
                radius=1.8,
            )
            ax_pie.set_aspect("equal")
        else:
            ax_pie.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_pie.set_aspect('equal')
        ax_pie.set_title("Share of Labeled Data Points")

        self.fig.tight_layout()
        self.canvas.draw()
