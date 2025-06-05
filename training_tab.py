###############################################################################
# training_tab_pro_plus.py
# ─────────────────────────────────────────────────────────────────────────────
# Multisensory Road-Condition Analysis – Training-Tab (Pro +  große Tabelle)
#
# Abhängigkeiten:
#   PyQt5, matplotlib, numpy, pandas, scikit-learn, torch
#   (getestet unter Python ≥3.9)
###############################################################################
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QLabel, QListWidget, QPushButton, QProgressBar,
    QSpinBox, QDoubleSpinBox, QTextBrowser, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QWidget, QSplitter, QFormLayout, QFrame, QSlider,
    QHeaderView
)

# ───────────────────────────────────────────  Matplotlib Setup ──
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

plt.rcParams.update({
    "toolbar": "None",
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
# ────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════
# Utility-Funktionen
# ════════════════════════════════════════════════════════════════
def _compute_oob_threshold(max_probs: np.ndarray, keep: float = 0.95) -> float:
    """Schwellwert τ für -OOB-Erkennung."""
    return float(np.quantile(max_probs, 1.0 - keep))


def _nice_number(n: int) -> str:
    """Formatiert große Zahlen mit schmalem Leerzeichen-Tausendertrenner."""
    return f"{n:,}".replace(",", " ")   # schmaler Space (U+00A0)


# ════════════════════════════════════════════════════════════════
# Train-Thread  (Daten laden, Netz trainieren, Ergebnisse emitten)
# ════════════════════════════════════════════════════════════════
class TrainWorker(QThread):
    # --------------------------- Signale ------------------------------------
    progress = pyqtSignal(int)                                  # 0-100 %
    step = pyqtSignal(int, float, float)                        # epoch, train-loss, val-loss
    log = pyqtSignal(str)
    finished = pyqtSignal(
        str,          # rep_str
        dict,         # rep_dict
        float,        # tau
        object,       # cm  (np.ndarray)
        object,       # class_weights (np.ndarray)
        dict,         # idx_to_name
        int,          # n_train
        int           # n_val
    )
    def __init__(
        self,
        folder: Path,
        label_map: Dict[str, int],
        unknown_id: int,
        *,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-5,
        patience: int = 5,
        keep_known: float = 0.95,
    ) -> None:
        super().__init__()
        self.folder, self.label_map = folder, label_map
        self.unknown_id = unknown_id
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.weight_decay, self.patience, self.keep_known = weight_decay, patience, keep_known

    # ----------------------------------------------------------------------
    def _load_csv_folder(self) -> Tuple[np.ndarray, np.ndarray]:
        """Alle CSVs im Ordner einlesen → X, y."""
        dfs: List[pd.DataFrame] = []
        csv_files = [
            f for f in self.folder.rglob("*.csv")
            if not f.name.endswith("_track.csv")
        ]
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue

            # Label-Spalte finden
            if "label_id" in df.columns:
                y = df["label_id"].fillna(self.unknown_id)
            elif "classification" in df.columns:
                y = df["classification"].fillna(self.unknown_id)
            elif "label_name" in df.columns:
                y = df["label_name"].map(self.label_map).fillna(self.unknown_id)
            else:
                continue

            feat_cols = [
                c for c in df.columns
                if c not in {"time", "label_id", "label_name", "classification"}
            ]
            X = df[feat_cols].copy()
            X["_y"] = y.astype(int)
            dfs.append(X)

        if not dfs:
            raise RuntimeError("No valid CSV files found.")

        data = pd.concat(dfs, ignore_index=True)
        y_orig = data.pop("_y").to_numpy(dtype=int)
        X = data.to_numpy()
        return X, y_orig

    # ----------------------------------------------------------------------
    def run(self) -> None:   # Haupt-Routine (läuft im Thread)
        try:
            X_raw, y_orig = self._load_csv_folder()

            # nur bekannte Klassen
            mask = y_orig != self.unknown_id
            X_known, y_known = X_raw[mask], y_orig[mask]
            n_train_samples = len(y_known)

            # ID <-> Idx Mapping
            unique_ids = sorted(set(y_known))
            id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
            idx_to_name = {
                i: next(k for k, v in self.label_map.items() if v == uid)
                for i, uid in enumerate(unique_ids)
            }
            y_idx = np.array([id_to_idx[v] for v in y_known], dtype=int)

            # Normalisierung
            scaler = StandardScaler()
            X_known = scaler.fit_transform(X_known)
            X_known = np.nan_to_num(X_known, nan=0.0)

            # Class-Weights für unbalancierte Daten
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_idx), y=y_idx
            ).astype(np.float32)

            # Train / Val-Split
            Xtr, Xte, ytr, yte = train_test_split(
                X_known, y_idx, test_size=0.2, stratify=y_idx, random_state=42
            )
            n_val_samples = len(yte)

            # Torch-Tensors
            Xtr, Xte = map(torch.FloatTensor, (Xtr, Xte))
            ytr, yte = map(torch.LongTensor, (ytr, yte))

            # Einfaches MLP
            model = nn.Sequential(
                nn.Linear(Xtr.shape[1], 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, len(unique_ids)),
            )
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

            loader = DataLoader(TensorDataset(Xtr, ytr),
                                batch_size=self.batch_size, shuffle=True)

            best_val, best_state, w_no_imp = np.inf, None, 0

            # ------------------------- Training-Loop ------------------------
            for ep in range(1, self.epochs + 1):
                model.train()
                tloss_sum = 0.0
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = crit(model(xb), yb)
                    loss.backward()
                    opt.step()
                    tloss_sum += float(loss) * len(xb)
                tloss = tloss_sum / len(Xtr)

                # Validation
                model.eval()
                with torch.no_grad():
                    vloss = crit(model(Xte), yte).item()

                self.step.emit(ep, tloss, vloss)
                self.progress.emit(int(ep / self.epochs * 100))

                # Early-Stopping
                if vloss < best_val - 1e-4:
                    best_val, best_state, w_no_imp = vloss, model.state_dict(), 0
                else:
                    w_no_imp += 1
                    if w_no_imp >= self.patience:
                        self.log.emit(f"Early stopping (epoch {ep})")
                        break

            if best_state:
                model.load_state_dict(best_state)

            # ------------------------- Evaluation ---------------------------
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(Xte), dim=1)
            conf, preds_idx = probs.max(1)

            tau = _compute_oob_threshold(conf.numpy(), keep=self.keep_known)

            y_true_ids = [unique_ids[int(i)] for i in yte]
            y_pred_ids = [unique_ids[int(i)] for i in preds_idx]

            rep_str = classification_report(
                y_true_ids, y_pred_ids, zero_division=0
            )
            rep_dict = classification_report(
                y_true_ids, y_pred_ids, zero_division=0, output_dict=True
            )

            cm = confusion_matrix(yte, preds_idx).astype(int)

            # Signal „finished“
            self.finished.emit(
                rep_str, rep_dict, tau, cm, class_weights, idx_to_name,
                n_train_samples, n_val_samples
            )

        except Exception as exc:   # Fehler → trotzdem senden
            self.finished.emit(
                f"Error: {exc}", {}, 0.0, np.zeros((1, 1), dtype=int),
                np.zeros(1), {}, 0, 0
            )


# ════════════════════════════════════════════════════════════════
# Matplotlib-Canvas für Live-Loss & Konfusionsmatrix
# ════════════════════════════════════════════════════════════════
class LiveLossCanvas(FigureCanvas):
    """Dynamischer Train/Val-Loss-Plot."""

    def __init__(self, parent=None):
        fig, self.ax = plt.subplots()
        super().__init__(fig)
        self.setParent(parent)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.tr_line, = self.ax.plot([], [], label="Train")
        self.val_line, = self.ax.plot([], [], label="Val")
        self.ax.legend()
        self.tr_vals: List[float] = []
        self.val_vals: List[float] = []

    # ------------------------------------------------------------------
    def update(self, epoch: int, train: float, val: float):
        self.tr_vals.append(train)
        self.val_vals.append(val)
        self.tr_line.set_data(range(1, len(self.tr_vals) + 1), self.tr_vals)
        self.val_line.set_data(range(1, len(self.val_vals) + 1), self.val_vals)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()


class ConfMatCanvas(FigureCanvas):
    """Heatmap der Konfusionsmatrix (inkl. Colorbar)."""

    def __init__(self, cm: np.ndarray, idx_to_name: Dict[int, str], parent=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        super().__init__(fig)
        self.setParent(parent)

        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax.set_title("Confusion Matrix (val)")
        ticks = np.arange(len(idx_to_name))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(
            [idx_to_name[i][:8] for i in ticks], rotation=45, ha="right"
        )
        ax.set_yticklabels([idx_to_name[i][:8] for i in ticks])

        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j, i,
                cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        fig.tight_layout()


# ════════════════════════════════════════════════════════════════
# Haupt-GUI-Klasse
# ════════════════════════════════════════════════════════════════
class TrainingTab(QWidget):
    def __init__(
        self, label_map: Dict[str, int], unknown_id: int, parent=None
    ) -> None:
        super().__init__(parent)
        self.label_map, self.unknown_id = label_map, unknown_id

        # Default-Ordner
        default = Path.home() / "Desktop/anomaly-data-hs-merseburg"
        self.folder = default if default.exists() else Path.cwd()

        # Splitter (L: Einstellungen / Dateien, R: Plots & Ergebnisse)
        splitter = QSplitter(Qt.Horizontal, self)
        left, right = QWidget(), QWidget()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 2)

        main = QVBoxLayout(self)
        main.addWidget(splitter)

        # ───────────────────────── LEFT PANE ──────────────────────────────
        lv = QVBoxLayout(left)

        # Ordner-Zeile
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel(str(self.folder))
        self.lbl_folder.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        hl.addWidget(self.lbl_folder, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        hl.addWidget(btn_browse)
        lv.addLayout(hl)

        # Dateiliste
        self.list_files = QListWidget()
        lv.addWidget(self.list_files, 2)
        self._update_file_list()

        # Hyperparameter-Form
        param_box = QFormLayout()

        self.sp_epochs = QSpinBox(); self.sp_epochs.setRange(1, 500); self.sp_epochs.setValue(50)
        param_box.addRow("Epochs", self.sp_epochs)

        self.sp_lr = QDoubleSpinBox(); self.sp_lr.setDecimals(5); self.sp_lr.setRange(1e-5, 1.0); self.sp_lr.setValue(1e-3)
        param_box.addRow("Learning Rate", self.sp_lr)

        self.sp_batch = QSpinBox(); self.sp_batch.setRange(32, 4096); self.sp_batch.setValue(256)
        param_box.addRow("Batch Size", self.sp_batch)

        self.sp_wd = QDoubleSpinBox(); self.sp_wd.setDecimals(6); self.sp_wd.setRange(0.0, 0.1); self.sp_wd.setValue(1e-5)
        param_box.addRow("Weight Decay", self.sp_wd)

        self.sp_pat = QSpinBox(); self.sp_pat.setRange(1, 20); self.sp_pat.setValue(5)
        param_box.addRow("Early-Stop Patience", self.sp_pat)

        # NEU: Slider für Tabellen-Schriftgröße
        self.sl_font = QSlider(Qt.Horizontal); self.sl_font.setRange(8, 18); self.sl_font.setValue(11)
        self.sl_font.valueChanged.connect(self._set_table_font)
        param_box.addRow("Table Font Size", self.sl_font)

        lv.addLayout(param_box)

        # Train-Button & Progress
        self.btn_train = QPushButton("Train model")
        self.btn_train.clicked.connect(self._train)
        lv.addWidget(self.btn_train)

        self.progress = QProgressBar()
        lv.addWidget(self.progress)

        # ───────────────────────── RIGHT PANE ─────────────────────────────
        rv = QVBoxLayout(right)

        # Val-Klassenverteilung (Balken)
        self.dist_canvas = FigureCanvas(plt.figure(figsize=(4, 2)))
        rv.addWidget(self.dist_canvas, 1)

        # Live-Loss
        self.live_canvas = LiveLossCanvas()
        rv.addWidget(self.live_canvas, 2)

        # Platz für Konfusionsmatrix
        self.cm_container = QVBoxLayout()
        rv.addLayout(self.cm_container, 2)

        # Ergebnis-Tabelle
        self.table = QTableWidget()
        self.table.setFont(QFont("DejaVu Sans Mono", 11))
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        rv.addWidget(self.table, 2)

        # Status-Label (aktive Klassen, Samples, τ)
        self.lbl_stats = QLabel()
        self.lbl_stats.setStyleSheet("font-weight:bold;")
        rv.addWidget(self.lbl_stats)

        # Log-Fenster
        self.txt = QTextBrowser()
        rv.addWidget(self.txt, 1)

    # ────────────────────────── Helper Slots ─────────────────────────────
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select CSV Folder", str(self.folder)
        )
        if path:
            self.folder = Path(path)
            self.lbl_folder.setText(path)
            self._update_file_list()

    def _update_file_list(self) -> None:
        self.list_files.clear()
        if not self.folder.exists():
            return
        for ext in ("*.csv", "*.json"):
            for f in sorted(self.folder.rglob(ext)):
                try:
                    self.list_files.addItem(str(f.relative_to(self.folder)))
                except ValueError:
                    self.list_files.addItem(f.name)

    # ────────────────────────── Training Start ───────────────────────────
    def _train(self) -> None:
        self.btn_train.setEnabled(False)
        self.progress.reset()

        # Visuals zurücksetzen
        self.live_canvas.tr_vals.clear()
        self.live_canvas.val_vals.clear()
        self.live_canvas.ax.cla()
        self.live_canvas.ax.set_xlabel("Epoch")
        self.live_canvas.ax.set_ylabel("Loss")
        self.live_canvas.tr_line, = self.live_canvas.ax.plot([], [], label="Train")
        self.live_canvas.val_line, = self.live_canvas.ax.plot([], [], label="Val")
        self.live_canvas.ax.legend()
        self.live_canvas.draw_idle()

        while self.cm_container.count():
            w = self.cm_container.takeAt(0).widget()
            if w is not None:
                w.setParent(None)

        self.lbl_stats.clear()
        self.txt.clear()
        self.table.clear()

        # Worker-Thread
        self.worker = TrainWorker(
            self.folder,
            self.label_map,
            self.unknown_id,
            epochs=self.sp_epochs.value(),
            lr=self.sp_lr.value(),
            batch_size=self.sp_batch.value(),
            weight_decay=self.sp_wd.value(),
            patience=self.sp_pat.value(),
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self._append)
        self.worker.step.connect(self._update_loss_plot)
        self.worker.finished.connect(self._done)
        self.worker.start()

        self._append(
            f"Training started with epochs={self.sp_epochs.value()}  "
            f"lr={self.sp_lr.value():.5f}  batch={self.sp_batch.value()}  "
            f"wd={self.sp_wd.value():.6f}"
        )

    # ------------------------------------------------------------------
    def _update_loss_plot(self, ep: int, train: float, val: float):
        self.live_canvas.update(ep, train, val)

    # ------------------------------------------------------------------
    def _done(
        self,
        rep_str: str,
        rep_dict: dict,
        tau: float,
        cm: np.ndarray,
        class_w: np.ndarray,
        idx_to_name: Dict[int, str],
        n_train: int,
        n_val: int,
    ):
        self.progress.setValue(100)
        self._append(rep_str)
        self._populate_table(rep_dict)

        self.lbl_stats.setText(
            f"Active classes: {len(idx_to_name)}   "
            f"Train: {_nice_number(n_train)}   "
            f"Val: {_nice_number(n_val)}   "
            f"τ = {tau:.3f}"
        )

        # Konfusionsmatrix
        cm_canvas = ConfMatCanvas(cm, idx_to_name)
        self.cm_container.addWidget(cm_canvas)
        cm_canvas.draw()

        # Val-Verteilung
        counts = cm.sum(axis=1)
        ax = self.dist_canvas.figure.clf().add_subplot(111)
        classes = [idx_to_name[i] for i in range(len(idx_to_name))]
        ax.bar(classes, counts)
        ax.set_title("Validation sample count per class")
        ax.set_xticklabels(classes, rotation=45, ha="right")
        self.dist_canvas.draw_idle()

        self.btn_train.setEnabled(True)

        # Hyperparameter-Summary
        self._append(
            "Hyperparameters:\n"
            f"epochs={self.sp_epochs.value()}  lr={self.sp_lr.value():.5f}  "
            f"batch={self.sp_batch.value()}  wd={self.sp_wd.value():.6f}  "
            f"class_weights={class_w.round(2).tolist()}"
        )

    # ------------------------------------------------------------------
    def _append(self, text: str):
        """Text ins Log."""
        self.txt.append(text)

    # ------------------------------------------------------------------
    def _populate_table(self, rep: dict):
        if not rep:
            return
        headers = ["Class", "Prec", "Recall", "F1", "Support"]
        rows = [k for k in rep.keys() if isinstance(rep[k], dict)]

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))

        for r, cls in enumerate(rows):
            d = rep[cls]
            vals = [
                str(cls),
                f"{d['precision']:.2f}",
                f"{d['recall']:.2f}",
                f"{d['f1-score']:.2f}",
                _nice_number(int(d['support'])),
            ]
            for c, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def _set_table_font(self, size: int):
        font = QFont("DejaVu Sans Mono", size)
        self.table.setFont(font)
        self.table.verticalHeader().setDefaultSectionSize(int(size * 2.2))


# ════════════════════════════════════════════════════════════════
# Entry-Point
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # Beispiel-Label-Mapping (anpassen / laden wie benötigt)
    dummy_labels = {
        "normal": 1,
        "depressed": 2,
        "cover": 3,
        "cobble": 4,
        "transverse": 5,
        "gravel": 6,
        "cracked": 7,
        "bump": 8,
        "uneven": 9,
        "damaged": 10,
    }
    unknown_id = 0

    app = QApplication(sys.argv)
    win = TrainingTab(dummy_labels, unknown_id)
    win.setWindowTitle("Multisensory Road Condition Analysis – Training")
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec_())
