#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relabel & Segment-Exporter  –  Zeitbasiertes Zuordnen der Peaks-Bilder
(inkl. Ausschluss bereits exportierter Segment-CSVs)
"""

from __future__ import annotations
import re, shutil, unicodedata
from pathlib import Path
from typing  import Dict, List, Tuple
import pandas as pd
from PyQt5.QtCore import Qt, QCoreApplication, QEventLoop
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGridLayout, QLabel, QPushButton, QTextEdit,
    QWidget, QCheckBox, QProgressBar,
)

# ------------------------------------------------------------------ Konstanten
CSV_RENAME: Dict[str, str] = {
    "transverse grove":          "curbs",
    "cracked/irregular surface": "cracked/ raveled roads",
}
PEAKS_PATTERN: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"transverse[\s_]+grove", re.I), "curbs"),
    (re.compile(
        r"cracked[\s_/]*[-]?[\s_/]*(irregular|raveled)[\s_/]+"
        r"(surface|pavement|roads|aspahlt)", re.I),
     "cracked_ raveled roads"),
]
UNKNOWN_ID     = 99
UNKNOWN_NAME   = "unknown"
SEGMENTS_ROOT  = "segments"
TIME_TOLERANCE = 0.2                       # Sekunden Puffer
SEG_RX         = re.compile(r"_seg\d{3}\.csv$", re.I)  # → Segment-Dateien

# ------------------------------------------------------------------ Hilfsfunktionen
def safe_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = text.strip().replace(" ", "_")
    return re.sub(r"__+", "_", text) or "unnamed"

def parse_prefix_time(name: str) -> float | None:
    """gibt float-Zeit zurück, wenn Name mit <zahl>_ beginnt"""
    m = re.match(r"^(\d+(?:\.\d+)?)_", name)
    return float(m.group(1)) if m else None

# ------------------------------------------------------------------ GUI-Klasse
class Relabeler(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Relabel & Segment-Exporter (time-based peaks)")
        self.resize(900, 610)

        grid = QGridLayout(self)
        self.lblDir = QLabel("Kein Ordner gewählt")
        self.btnSel = QPushButton("Hauptordner wählen …")
        self.chkDbg = QCheckBox("Debug")
        self.bar    = QProgressBar()
        self.bar.setTextVisible(True)
        self.log    = QTextEdit(readOnly=True)

        grid.addWidget(self.lblDir, 0, 0)
        grid.addWidget(self.btnSel, 0, 1)
        grid.addWidget(self.chkDbg, 0, 2)
        grid.addWidget(self.bar,    0, 3)
        grid.addWidget(self.log,    1, 0, 1, 4)

        self.btnSel.clicked.connect(self.choose_folder)

    # ========== Ordner wählen =================================================
    def choose_folder(self) -> None:
        root = QFileDialog.getExistingDirectory(
            self, "Root-Ordner wählen", str(Path.home())
        )
        if root:
            self.lblDir.setText(root)
            self.log.clear()
            self.run(Path(root))

    # ========== Hauptablauf ===================================================
    def run(self, root: Path) -> None:
        self.setWindowTitle("Bearbeitung läuft …")

        # --- CSV-Liste (Originale) -------------------------------------------
        csvs = [
            p for p in root.rglob("*.csv")
            if (SEGMENTS_ROOT not in p.parts)          # nix unter ./segments
               and not p.name.endswith("_track.csv")   # keine *_track.csv
               and not SEG_RX.search(p.name)           # keine *_seg###.csv
        ]
        # --- Peaks-Unterordner (zum UMBENENNEN) ------------------------------
        peaks_dirs = [
            d for p in root.rglob("peaks") if p.is_dir()
            for d in p.iterdir() if d.is_dir()
        ]

        self.bar.setMaximum(len(csvs) + len(peaks_dirs) or 1)

        csv_log, dir_log, seg_log, img_log = [], [], [], []
        seg_root = root / SEGMENTS_ROOT
        seg_root.mkdir(exist_ok=True)

        # ---------- CSV-Verarbeitung -----------------------------------------
        for i, csv in enumerate(csvs, 1):
            self._dbg(f"CSV {csv.relative_to(root)}")
            changed, df = self._fix_csv(csv, csv_log, root)
            if df is not None:
                self._export_segments(df, csv, seg_root, root, seg_log, img_log)
            if df is not None and not changed:
                self._log(f"[OK]  {csv.relative_to(root)}: keine Label-Änderung")
            self._tick(i)

        # ---------- Peaks-Ordner umbenennen -----------------------------------
        for j, sub in enumerate(peaks_dirs, 1):
            self._rename_peak_dir(sub, dir_log, root)
            self._tick(len(csvs) + j)

        # ---------- Zusammenfassung ------------------------------------------
        for title, items in [
            ("CSV geändert",   csv_log),
            ("Peaks umbenannt", dir_log),
            ("Segment-CSVs",   seg_log),
            ("Peaks kopiert",  img_log),
        ]:
            self._summary(title, items)

        self.setWindowTitle("Fertig")

    # ---------------------------------------------------------------- Fortschritt
    def _tick(self, val: int) -> None:
        self.bar.setValue(val)
        QCoreApplication.processEvents(QEventLoop.AllEvents, 5)

    # ---------------------------------------------------------------- CSV-Anpassung
    def _fix_csv(
        self, csv: Path, clog: list[str], root: Path
    ) -> tuple[bool, pd.DataFrame | None]:
        try:
            df = pd.read_csv(csv)
        except Exception as err:
            self._log(f"[ERR] {csv.relative_to(root)}: {err}")
            return False, None

        if "label_name" not in df.columns:
            return False, df

        counter = 0
        for old, new in CSV_RENAME.items():
            mask = df["label_name"].str.strip().str.casefold() == old.casefold()
            counter += mask.sum()
            df.loc[mask, "label_name"] = new

        if counter:
            try:
                df.to_csv(csv, index=False)
                clog.append(f"{csv.relative_to(root)} ({counter})")
            except Exception as err:
                self._log(f"[ERR] save {csv.relative_to(root)}: {err}")
        return bool(counter), df

    # ---------------------------------------------------------------- Segmente + Peaks
    def _export_segments(
        self, df: pd.DataFrame, src: Path, seg_root: Path, root: Path,
        seg_log: list[str], img_log: list[str]
    ) -> None:
        seg_id = (df["label_name"] != df["label_name"].shift()).cumsum()
        peaks_base = src.parent / "peaks"

        for idx, grp in df.groupby(seg_id):
            lname = str(grp["label_name"].iloc[0]).strip()
            lid   = int(grp["label_id"].iloc[0]) if "label_id" in grp else None
            if lid == UNKNOWN_ID or lname.casefold() == UNKNOWN_NAME:
                continue

            t0, t1 = float(grp["time"].iloc[0]), float(grp["time"].iloc[-1])
            tgt_label = seg_root / safe_name(lname)
            tgt_label.mkdir(parents=True, exist_ok=True)

            csv_out = tgt_label / f"{src.stem}_seg{idx:03d}.csv"
            try:
                grp.to_csv(csv_out, index=False)
                seg_log.append(str(csv_out.relative_to(root)))
            except Exception as err:
                self._log(f"[ERR] write {csv_out.relative_to(root)}: {err}")

            self._copy_time_peaks(
                peaks_base, t0, t1,
                tgt_label / "peaks" / f"seg{idx:03d}",
                root, img_log
            )

    # ---------------------------------------------------------------- Peaks kopieren
    def _copy_time_peaks(
        self, base: Path, t0: float, t1: float,
        dest: Path, root: Path, img_log: list[str]
    ) -> None:
        if not base.exists():
            return

        images: list[Path] = []
        for d in base.iterdir():
            if not d.is_dir():
                continue
            tt = parse_prefix_time(d.name)
            if tt is None:
                continue
            if t0 - TIME_TOLERANCE <= tt <= t1 + TIME_TOLERANCE:
                images.extend(d.glob("*.png"))

        self._dbg(f"Peaks {t0:.2f}-{t1:.2f}: {len(images)} Bilder")

        if not images:
            return

        dest.mkdir(parents=True, exist_ok=True)
        for img in images:
            try:
                shutil.copy2(img, dest / img.name)
                img_log.append(str((dest / img.name).relative_to(root)))
            except Exception as err:
                self._log(f"[ERR] copy {img.relative_to(root)}: {err}")

    # ---------------------------------------------------------------- Peaks-Ordner-Rename
    def _rename_peak_dir(self, sub: Path, dlog: list[str], root: Path) -> None:
        old, new = sub.name, sub.name
        for pat, repl in PEAKS_PATTERN:
            if pat.search(old):
                parts = old.split("_", 1)
                prefix = parts[0] + "_" if len(parts) > 1 else ""
                new = prefix + pat.sub(repl, parts[-1])
                break

        if new != old:
            target = sub.with_name(new)
            if target.exists():
                self._log(f"[WARN] {sub.relative_to(root)} → Ziel existiert")
                return
            try:
                sub.rename(target)
                dlog.append(f"{sub.relative_to(root)} → {target.name}")
            except Exception as err:
                self._log(f"[ERR] rename {sub.relative_to(root)}: {err}")

    # ---------------------------------------------------------------- Logging / Debug
    def _log(self, txt: str) -> None:
        self.log.append(txt)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )

    def _dbg(self, txt: str) -> None:
        if self.chkDbg.isChecked():
            self._log("[DBG] " + txt)

    def _summary(self, title: str, items: List[str]) -> None:
        self._log("")  # Leerzeile
        if items:
            self._log(f"{title}:")
            self._log("  • " + "\n  • ".join(items))
        else:
            self._log(f"{title}: —")

# ------------------------------------------------------------------ main()
if __name__ == "__main__":
    import sys
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # High-DPI-Fix
    app = QApplication(sys.argv)
    gui = Relabeler()
    gui.show()
    sys.exit(app.exec_())
