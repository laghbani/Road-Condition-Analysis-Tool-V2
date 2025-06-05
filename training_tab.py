from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QProgressBar,
    QTextBrowser,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
)
from PyQt5.QtCore import QThread, pyqtSignal


class TrainWorker(QThread):
    """Background worker that trains a simple model."""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str, dict)

    def __init__(
        self,
        folder: Path,
        label_map: Dict[str, int],
        unknown_id: int,
        *,
        epochs: int = 20,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.label_map = label_map
        self.unknown_id = unknown_id
        self.epochs = epochs
        self.lr = lr

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            dfs = []
            csv_files = [f for f in self.folder.rglob("*.csv") if not f.name.endswith("_track.csv")]
            for csv in csv_files:
                try:
                    df = pd.read_csv(csv)
                except Exception:
                    continue
                if "label_id" in df.columns:
                    y = df["label_id"].fillna(self.unknown_id)
                elif "classification" in df.columns:
                    y = df["classification"].fillna(self.unknown_id)
                elif "label_name" in df.columns:
                    y = df["label_name"].map(self.label_map).fillna(self.unknown_id)
                else:
                    continue
                feat_cols = [c for c in df.columns if c not in {"time", "label_id", "label_name", "classification"}]
                X = df[feat_cols].copy()
                X.loc[:, "_y"] = y.astype(int)
                dfs.append(X)

            if not dfs:
                self.finished.emit("No valid CSV files found.")
                return

            data = pd.concat(dfs, ignore_index=True)
            y_orig = data.pop("_y").to_numpy().astype(int)
            X = data.to_numpy()

            # map original IDs (incl. unknown) to contiguous indices
            id_to_name = {v: k for k, v in self.label_map.items()}
            id_to_name[self.unknown_id] = "unknown"
            unique_ids = sorted(id_to_name.keys())
            id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
            idx_to_name = {i: id_to_name[uid] for uid, i in id_to_idx.items()}
            y = np.array([id_to_idx.get(v, id_to_idx[self.unknown_id]) for v in y_orig], dtype=int)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            if not np.isfinite(X).all():
                self.log.emit("⚠️ Invalid numeric values detected. Replacing NaNs/Infs with zero.")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            unique, counts = np.unique(y_orig, return_counts=True)
            class_info = {uid: cnt for uid, cnt in zip(unique, counts)}
            self.log.emit(f"Samples: {len(X)}, Features: {X.shape[1]}")
            for uid, cnt in class_info.items():
                name = id_to_name.get(uid, str(uid))
                self.log.emit(f"Class {name}: {cnt} samples")
                if cnt < 50:
                    self.log.emit(f"⚠️ More data recommended for class '{name}'")

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            Xtr = torch.tensor(Xtr, dtype=torch.float32)
            Xte = torch.tensor(Xte, dtype=torch.float32)
            ytr = torch.tensor(ytr, dtype=torch.long)
            yte = torch.tensor(yte, dtype=torch.long)

            model = nn.Sequential(
                nn.Linear(Xtr.shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, len(id_to_idx))
            )
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log.emit(f"Model parameters: {params}")
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            crit = nn.CrossEntropyLoss()

            loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)

            for ep in range(1, self.epochs + 1):
                tloss = []
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = crit(model(xb), yb)
                    loss.backward()
                    opt.step()
                    tloss.append(float(loss))

                with torch.no_grad():
                    vloss = crit(model(Xte), yte).item()
                self.log.emit(f"Epoch {ep}/{self.epochs} loss={np.mean(tloss):.4f} val={vloss:.4f}")
                self.progress.emit(int(ep / self.epochs * 100))

            with torch.no_grad():
                preds_idx = model(Xte).argmax(1).numpy()
            preds_orig = [unique_ids[i] for i in preds_idx]
            true_orig = [unique_ids[i] for i in yte.numpy()]
            rep_str = classification_report(true_orig, preds_orig, zero_division=0)
            rep_dict = classification_report(true_orig, preds_orig, zero_division=0, output_dict=True)
            self.finished.emit(rep_str, rep_dict)
        except Exception as exc:  # pragma: no cover - just log
            self.finished.emit(f"Error: {exc}", {})


class TrainingTab(QWidget):
    """UI for loading data and training a small model."""

    def __init__(self, label_map: Dict[str, int], unknown_id: int, parent=None) -> None:
        super().__init__(parent)
        self.label_map = label_map
        self.unknown_id = unknown_id
        default = Path.home() / "Desktop/anomaly-data-hs-merseburg"
        self.folder = default if default.exists() else Path.cwd()

        vbox = QVBoxLayout(self)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel(str(self.folder))
        hl.addWidget(self.lbl_folder, 1)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        hl.addWidget(self.btn_browse)
        vbox.addLayout(hl)

        self.list_files = QListWidget()
        vbox.addWidget(self.list_files)
        self._update_file_list()

        par = QHBoxLayout()
        par.addWidget(QLabel("Epochs:"))
        self.sp_epochs = QSpinBox()
        self.sp_epochs.setRange(1, 1000)
        self.sp_epochs.setValue(20)
        par.addWidget(self.sp_epochs)
        par.addWidget(QLabel("LR:"))
        self.sp_lr = QDoubleSpinBox()
        self.sp_lr.setDecimals(5)
        self.sp_lr.setRange(1e-5, 1.0)
        self.sp_lr.setValue(1e-3)
        par.addWidget(self.sp_lr)
        vbox.addLayout(par)

        self.btn_train = QPushButton("Train model")
        self.btn_train.clicked.connect(self._train)
        vbox.addWidget(self.btn_train)

        self.progress = QProgressBar()
        vbox.addWidget(self.progress)

        self.txt = QTextBrowser()
        vbox.addWidget(self.txt, 1)

        self.table = QTableWidget()
        vbox.addWidget(self.table)

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select CSV Folder", str(self.folder))
        if path:
            self.folder = Path(path)
            self.lbl_folder.setText(path)
            self._update_file_list()

    def _train(self) -> None:
        self.btn_train.setEnabled(False)
        self.progress.setValue(0)
        self.txt.clear()
        self.table.clear()
        self._update_file_list()
        self.worker = TrainWorker(
            self.folder,
            self.label_map,
            self.unknown_id,
            epochs=self.sp_epochs.value(),
            lr=self.sp_lr.value(),
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self._append)
        self.worker.finished.connect(self._done)
        self.worker.start()
        self._append(
            f"Training for {self.sp_epochs.value()} epochs at lr={self.sp_lr.value():.5f}"
        )

    def _update_file_list(self) -> None:
        """Display CSV and JSON files in the selected folder."""
        self.list_files.clear()
        if not self.folder.exists():
            return
        for ext in ("*.csv", "*.json"):
            for file in sorted(self.folder.rglob(ext)):
                try:
                    self.list_files.addItem(str(file.relative_to(self.folder)))
                except ValueError:
                    self.list_files.addItem(file.name)

    def _done(self, rep_str: str, rep_dict: dict) -> None:
        self.progress.setValue(100)
        self._append(rep_str)
        self._populate_table(rep_dict)
        self.btn_train.setEnabled(True)

    def _append(self, text: str) -> None:
        self.txt.append(text)

    def _populate_table(self, rep: dict) -> None:
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
                cls,
                f"{d['precision']:.2f}",
                f"{d['recall']:.2f}",
                f"{d['f1-score']:.2f}",
                str(d['support']),
            ]
            for c, val in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(val))
        self.table.resizeColumnsToContents()
