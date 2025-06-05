from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib as mpl
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
)

# ------------------------------------------------------------
# ein paar dezente globale Stilvorgaben
# ------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6,
})

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # Matplotlib < 3.8
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class StatsTab(QWidget):
    """Visualisiert Daten-, Label- und Gruppenzahlen als Pie- und Balkendiagramme."""

    def __init__(self, colors: dict[str, str], unknown_name: str, parent=None) -> None:
        super().__init__(parent)

        # ---------- Layout der ganzen QWidget-Seite ----------
        vbox = QVBoxLayout(self)

        # -- 1) obere Steuerleiste ------------------------------------------------
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

        # -- 2) Zeile mit Gesamt-Zahlen ------------------------------------------
        hl2 = QHBoxLayout()
        self.lbl_totals = QLabel("")
        hl2.addWidget(self.lbl_totals)
        hl2.addStretch()
        vbox.addLayout(hl2)

        # -- 3) große Zeichenfläche ----------------------------------------------
        #   • insgesamt breiter (=14 statt 12 Zoll)
        #   • constrained layout verhindert Überlappungen
        self.fig = Figure(figsize=(14, 8), dpi=100, layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, stretch=1)

        # ------------------------------------------------------------------------
        self.colors = colors
        self.unknown_name = unknown_name

    # ============================================================================
    # Dateiauswahl & PDF-Export
    # ============================================================================
    def _browse(self) -> None:
        start_dir = "/home/afius/Desktop/anomaly-data-hs-merseburg"
        path = QFileDialog.getExistingDirectory(
            self, "Select CSV Folder", start_dir
        )
        if path:
            self.load_folder(path)

    def _save_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot as PDF", "", "PDF Files (*.pdf)"
        )
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self.fig.savefig(path)

    # ============================================================================
    # CSVs einlesen & Statistiken visualisieren
    # ============================================================================
    def load_folder(self, folder: str | Path) -> None:
        folder = Path(folder)
        self.lbl_path.setText(str(folder))

        csv_files = [
            f for f in folder.rglob("*.csv")
            if not f.name.endswith("_track.csv")
        ]

        dp_counts = defaultdict(int)   # Anzahl Datenpunkte pro Klasse
        grp_counts = defaultdict(int)  # Anzahl Gruppen pro Klasse

        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if "label_name" not in df.columns:
                continue

            labels = df["label_name"].fillna(self.unknown_name)

            # Datenpunkte zählen
            for lbl, cnt in labels.value_counts().items():
                dp_counts[str(lbl)] += int(cnt)

            # Gruppen zählen
            seg_id = (labels != labels.shift()).cumsum()
            for _, grp in labels.groupby(seg_id):
                grp_counts[str(grp.iloc[0])] += 1

        total   = sum(dp_counts.values())
        unknown = dp_counts.get(self.unknown_name, 0)
        labeled = total - unknown
        self.lbl_totals.setText(
            f"Total: {total} | Labeled: {labeled} | Unknown: {unknown}"
        )

        self._plot_stats(dp_counts, grp_counts)

    # ----------------------------------------------------------------------------
    # Plot-Routine
    # ----------------------------------------------------------------------------
    def _plot_stats(
        self,
        dp_counts: dict[str, int],
        grp_counts: dict[str, int],
    ) -> None:
        self.fig.clear()

        if not dp_counts and not grp_counts:
            self.canvas.draw()
            return

        # -- Daten vorbereiten ---------------------------------------------------
        labels = [
            l for l in sorted(set(dp_counts) | set(grp_counts))
            if l != self.unknown_name
        ]
        dp_vals  = [dp_counts.get(l, 0)  for l in labels]
        grp_vals = [grp_counts.get(l, 0) for l in labels]
        cols     = [self.colors.get(l, "#808080") for l in labels]

        x = range(len(labels))
        width = 0.6

        # -- Layout: kleineres Pie (Spalte 0) | breite Balkendiagramme (Spalte 1)
        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[1, 1.8],   # << rechts ~80 % breiter als links
            wspace=0.15,             # etwas Abstand in x-Richtung
        )

        ax_pie = self.fig.add_subplot(gs[:, 0])
        ax_dp  = self.fig.add_subplot(gs[0, 1])
        ax_grp = self.fig.add_subplot(gs[1, 1], sharex=ax_dp)

        # ---------- Balkendiagramm 1: Data Points --------------------------------
        bar_dp = ax_dp.bar(
            x, dp_vals, width=width,
            color=cols, edgecolor="#333333", linewidth=0.6
        )
        ax_dp.set_ylabel("Data Points")
        ax_dp.set_title("Data Points per Class", pad=6)
        ax_dp.grid(axis="y", alpha=0.25)

        # ***  x-Achse oben AUSblenden  ****
        ax_dp.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # ---------- Balkendiagramm 2: Groups -------------------------------------
        bar_grp = ax_grp.bar(
            x, grp_vals, width=width,
            color=cols, edgecolor="#333333", linewidth=0.6
        )
        ax_grp.set_ylabel("Groups")
        ax_grp.set_title("Groups per Class", pad=6)
        ax_grp.grid(axis="y", alpha=0.25)
        ax_grp.set_xticks(x)
        ax_grp.set_xticklabels(labels, rotation=35, ha="right")
        ax_grp.bar_label(bar_grp, fmt="%.0f", padding=3, fontsize=8)

        # ---------- Kreisdiagramm -----------------------------------------------
        if sum(dp_vals) > 0:
            wedges, texts, autotexts = ax_pie.pie(
                dp_vals,
                labels=None,            # nur Prozentwerte
                colors=cols,
                autopct="%1.1f%%",
                startangle=140,
                radius=1.35,            # << kleinerer Radius
                pctdistance=0.8,
            )
            for autotext in autotexts:
                autotext.set_fontsize(9)
        else:
            ax_pie.text(
                0.5, 0.5, "No data",
                ha="center", va="center", fontsize=12, weight="bold"
            )

        ax_pie.set_aspect("equal")
        ax_pie.set_title("Share of Labeled Data Points", pad=6)

        # ------------------------------------------------------------------------
        self.canvas.draw()
