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
)
from PyQt5.QtCore import QThread, pyqtSignal


class TrainWorker(QThread):
    """Background worker that trains a simple model."""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

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
                X = df[feat_cols]
                X["_y"] = y.astype(int)
                dfs.append(X)

            if not dfs:
                self.finished.emit("No valid CSV files found.")
                return

            data = pd.concat(dfs, ignore_index=True)
            y = data.pop("_y").to_numpy()
            X = data.to_numpy()

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            unique, counts = np.unique(y, return_counts=True)
            class_info = {uid: cnt for uid, cnt in zip(unique, counts)}
            self.log.emit(f"Samples: {len(X)}, Features: {X.shape[1]}")
            for uid, cnt in class_info.items():
                name = next((n for n, i in self.label_map.items() if i == uid), str(uid))
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
                nn.Linear(128, len(np.unique(y)))
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
                preds = model(Xte).argmax(1).numpy()
            rep = classification_report(yte.numpy(), preds)
            self.finished.emit(rep)
        except Exception as exc:  # pragma: no cover - just log
            self.finished.emit(f"Error: {exc}")


class TrainingTab(QWidget):
    """UI for loading data and training a small model."""

    def __init__(self, label_map: Dict[str, int], unknown_id: int, parent=None) -> None:
        super().__init__(parent)
        self.label_map = label_map
        self.unknown_id = unknown_id
        self.folder = Path("/home/afius/Desktop/anomaly-data-hs-merseburg")

        vbox = QVBoxLayout(self)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel(str(self.folder))
        hl.addWidget(self.lbl_folder, 1)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        hl.addWidget(self.btn_browse)
        vbox.addLayout(hl)

        self.btn_train = QPushButton("Train model")
        self.btn_train.clicked.connect(self._train)
        vbox.addWidget(self.btn_train)

        self.progress = QProgressBar()
        vbox.addWidget(self.progress)

        self.txt = QTextBrowser()
        vbox.addWidget(self.txt, 1)

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select CSV Folder", str(self.folder))
        if path:
            self.folder = Path(path)
            self.lbl_folder.setText(path)

    def _train(self) -> None:
        self.btn_train.setEnabled(False)
        self.progress.setValue(0)
        self.txt.clear()
        self.worker = TrainWorker(self.folder, self.label_map, self.unknown_id)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self._append)
        self.worker.finished.connect(self._done)
        self.worker.start()

    def _done(self, msg: str) -> None:
        self.progress.setValue(100)
        self._append(msg)
        self.btn_train.setEnabled(True)

    def _append(self, text: str) -> None:
        self.txt.append(text)
