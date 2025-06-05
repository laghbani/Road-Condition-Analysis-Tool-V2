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

        total_counts = defaultdict(int)
        labeled_counts = defaultdict(int)
        grp_counts = defaultdict(int)

        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if "label_name" not in df.columns:
                continue
            labels = df["label_name"].fillna(self.unknown_name)

            # --- total vs labeled counts ---
            for lbl, cnt in labels.value_counts().items():
                total_counts[str(lbl)] += int(cnt)
                if str(lbl) != self.unknown_name:
                    labeled_counts[str(lbl)] += int(cnt)

            # --- group counts ---
            seg_id = (labels != labels.shift()).cumsum()
            for _, grp in labels.groupby(seg_id):
                grp_counts[str(grp.iloc[0])] += 1

        self._plot_stats(total_counts, labeled_counts, grp_counts)

    def _plot_stats(
        self,
        total_counts: dict[str, int],
        labeled_counts: dict[str, int],
        grp_counts: dict[str, int],
    ) -> None:
        self.fig.clear()
        if not total_counts and not grp_counts:
            self.canvas.draw()
            return

        all_labels = sorted(set(total_counts) | set(labeled_counts) | set(grp_counts))
        tot_vals = [total_counts.get(lbl, 0) for lbl in all_labels]
        lab_vals = [labeled_counts.get(lbl, 0) for lbl in all_labels]
        grp_vals = [grp_counts.get(lbl, 0) for lbl in all_labels]
        cols = [self.colors.get(lbl, "#808080") for lbl in all_labels]

        ax1 = self.fig.add_subplot(211)
        x = range(len(all_labels))
        width = 0.25
        ax1.bar([i - width for i in x], tot_vals, width=width, label="Total", color="lightgray")
        ax1.bar(x, lab_vals, width=width, label="Labeled", color=cols)
        ax1.bar([i + width for i in x], grp_vals, width=width, label="Groups", color="black", alpha=0.3)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(all_labels, rotation=45)
        ax1.set_ylabel("Count")
        ax1.set_title("Counts per Class")
        ax1.legend()

        ax2 = self.fig.add_subplot(212)
        if sum(lab_vals) > 0:
            ax2.pie(lab_vals, labels=all_labels, colors=cols, autopct="%1.1f%%", startangle=140)
        else:
            ax2.text(0.5, 0.5, "No labeled data", ha="center", va="center")
        ax2.set_title("Share of Labeled Data Points")

        self.fig.tight_layout()
        self.canvas.draw()
