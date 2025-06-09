#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
road_anomaly_trainer_hybrid_gpu_safe.py
=======================================

• GroupKFold pro CSV-Datei (kein Daten-Leakage)
• RandomOverSampler ("not majority") gleicht Klassen aus
• HybridNet: ConvStem → BiLSTM → Attention-Pooling
• Focal-Loss (γ = 1, Label-Smooth = 0.05), Gradient-Clipping
• Automatic Mixed Precision (AMP) + Pin-Memory
• CUDNN-Benchmarking und – falls GPU-VRAM voll läuft – automatischer
  Fallback auf CPU ohne Abbruch des Trainings
• Vollständiges Qt-GUI (Loss-Plot, Confusion-Matrix, PR-Kurven,
  Paket-Export: Gewichte + PDF-Bericht)
"""

# ──────────────────── BASIS-IMPORTS & LOGGING ────────────────────
from __future__ import annotations
import os, sys, itertools, logging, contextlib
from pathlib import Path
from typing import Dict, List, Tuple
import random
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, auc
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import RandomOverSampler

from matplotlib.backends.backend_pdf import PdfPages

# ───────── Optionales Time-Warp (nur wenn torchaudio vorhanden) ──
HAVE_AUDIO = False            # kann bei Bedarf aktiviert werden

# ──────────────────── GUI- & MPL-Setup ───────────────────────────
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSplitter, QFileDialog, QLabel, QPushButton,
    QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QTableWidget,
    QTableWidgetItem, QTextBrowser, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QFrame, QScrollArea, QHeaderView, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from numpy.fft import rfft

# ──────────────────── GPU-SETUP ──────────────────────────────────
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
INIT_DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
logging.getLogger().setLevel(logging.INFO)
logging.info(f"Start on {INIT_DEVICE}")

plt.rcParams.update({
    "toolbar": "None", "figure.dpi": 120,
    "axes.facecolor": "#fafafa", "axes.grid": True, "grid.alpha": .25,
    "font.size": 9,
})
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"        # Intel-MKL Workaround

# ═══════════════════ Hilfs-Funktionen & Augment ==================
def _nice(n: int) -> str: return f"{n:,}".replace(",", " ")

def add_gauss(x: np.ndarray, snr_db: float = 25) -> np.ndarray:
    pwr = np.mean(x ** 2)
    sigma = np.sqrt(pwr / 10 ** (snr_db / 10))
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)

def sign_flip(x: np.ndarray, p: float = .5) -> np.ndarray:
    if np.random.rand() < p: x[:3] *= -1
    return x

def scale_amplitude(x: np.ndarray, factor: float = 0.1) -> np.ndarray:
    scale = np.random.uniform(1 - factor, 1 + factor)
    return x * scale

def dropout_segment(x: np.ndarray, max_pct: float = 0.1) -> np.ndarray:
    """Zero out a random short segment to simulate sensor dropout."""
    if max_pct <= 0:
        return x
    L = x.shape[1]
    span = max(1, int(L * max_pct))
    start = np.random.randint(0, L - span + 1)
    x[:, start:start + span] = 0
    return x

def time_warp(x: np.ndarray, factor: float = .1) -> np.ndarray:
    if not HAVE_AUDIO: return x
    import torchaudio
    rate = 1 + np.random.uniform(-factor, factor)
    ten = torch.from_numpy(x.copy()).unsqueeze(0)
    warpd, _ = torchaudio.functional.time_stretch(ten, rate, n_freq=1)
    return warpd.squeeze(0).numpy()

# ═══════════════════ Datensatz-Klasse ============================
class IMUWindowDataset(Dataset):
    EXTRA_CANDS = [
        ("gyro_x", "gyro_y", "gyro_z"),
        ("|a|_raw",), ("|ω|_raw",),
    ]
    def __init__(self, csvs: List[Path], *, label_map: Dict[str, int],
                 window: int = 256, stride: int = 128, majority_thr: float = .7,
                 add_speed: bool = True, fft_bins: int = 0, speed_max: float = 6,
                 unknown_id: int = 99, augment: bool = False):
        super().__init__()
        self.X, self.y, self.grp = [], [], []
        L = window

        for csv in csvs:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if "label_id" not in df.columns:
                continue

            # Basis-Kanäle
            if {"accel_veh_x", "accel_veh_y", "accel_veh_z"}.issubset(df.columns):
                base = ["accel_veh_x", "accel_veh_y", "accel_veh_z"]
            elif {"accel_corr_x", "accel_corr_y", "accel_corr_z"}.issubset(df.columns):
                base = ["accel_corr_x", "accel_corr_y", "accel_corr_z"]
            else:
                base = ["accel_x", "accel_y", "accel_z"]
            cols: list[str] = list(base)
            for cand in self.EXTRA_CANDS:
                if set(cand).issubset(df.columns):
                    cols.extend(cand)
            if add_speed and "speed_mps" in df.columns:
                cols.append("speed_mps")

            arr = df[cols].to_numpy(np.float32)
            lbl = df["label_id"].to_numpy(np.int16)

            if add_speed and "speed_mps" in df.columns:
                idx_speed = cols.index("speed_mps")
                arr[:, idx_speed] = np.clip(arr[:, idx_speed] / speed_max, 0, 1)

            for s in range(0, len(df) - L + 1, stride):
                win_lbl = lbl[s:s+L]
                maj = np.bincount(win_lbl).argmax()
                if maj == unknown_id or (win_lbl == maj).mean() < majority_thr:
                    continue
                win = arr[s:s+L].T      # C × L
                if fft_bins:
                    specs = []
                    for c in range(3):
                        mag = np.abs(rfft(win[c]))[:fft_bins].astype(np.float32)
                        specs.append(np.repeat(mag[:, None], L, axis=1))
                    win = np.concatenate([win, *specs], 0)
                if augment:
                    win = sign_flip(add_gauss(scale_amplitude(dropout_segment(win))))
                self.X.append(win); self.y.append(int(maj)); self.grp.append(csv.stem)

        if not self.X:
            raise RuntimeError("Keine Fenster nach Filterung übrig.")

        self.X = np.stack(self.X).astype(np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        self.grp = np.array(self.grp)

        # Standard-Scaling kanalweise
        C = self.X.shape[1]
        flat = self.X.transpose(0, 2, 1).reshape(-1, C)
        flat = StandardScaler().fit_transform(flat)
        self.X = flat.reshape(-1, self.X.shape[2], C).transpose(0, 2, 1)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx]), int(self.y[idx])

# ═══════════════════ Modell-Blöcke ===============================
class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.w = nn.Linear(dim, 1, bias=False)
    def forward(self, x):               # x : B × L × D
        α = torch.softmax(self.w(x).squeeze(-1), 1)
        return (α.unsqueeze(-1) * x).sum(1)

class ConvStem(nn.Module):
    def __init__(self, c_in: int, c_out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, 7, padding=3, bias=False),
            nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),     # L / 2
        )
    def forward(self, x): return self.net(x)

class HybridNet(nn.Module):
    def __init__(self, c_in: int, n_cls: int,
                 hidden: int = 128, layers: int = 2, drop: float = .5):
        super().__init__()
        self.in_channels = c_in
        self.stem = ConvStem(c_in)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=drop if layers > 1 else 0.)
        self.att = AttentionPool(hidden*2)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hidden*2, n_cls)
    def forward(self, x):               # x : B × C × L
        x = self.stem(x)                # B × 64 × L/2
        x = x.transpose(1, 2)           # B × L/2 × 64
        out, _ = self.lstm(x)
        rep = self.dropout(self.att(out))
        return self.fc(rep)

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, c_in: int, n_cls: int, base: int = 64,
                 layers: tuple[int, int, int, int] = (2, 2, 2, 2)):
        super().__init__()
        self.inplanes = base
        self.conv1 = nn.Conv1d(c_in, base, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(base, layers[0])
        self.layer2 = self._make_layer(base*2, layers[1], stride=2)
        self.layer3 = self._make_layer(base*4, layers[2], stride=2)
        self.layer4 = self._make_layer(base*8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base*8, n_cls)
        self.in_channels = c_in
    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = [BasicBlock1D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ═══════════════════ Focal-Loss & Metriken =======================
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor,
                 gamma: float = 1.0, smooth: float = .05):
        super().__init__(); self.register_buffer("a", alpha)
        self.g, self.s = gamma, smooth
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        at = self.a[targets] if self.a.numel() > 1 else 1.
        pt = torch.exp(-ce)
        fl = at * (1-pt)**self.g * ce
        return (1-self.s)*fl.mean() + self.s*ce.mean()

def pr_curves(y_true, y_prob, n_cls):
    curves = {}
    for i in range(n_cls):
        p, r, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        curves[i] = (p, r, auc(r, p))
    return curves

def _oob_threshold(max_probs: np.ndarray, keep: float = .95) -> float:
    return float(np.quantile(max_probs, 1-keep))

# ═══════════════════ Matplotlib-Canvas – Loss ====================
class LossCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(6, 2))
        super().__init__(fig); self.setParent(parent)
        self.ax.set(title="Loss-Verlauf", xlabel="Epoch",
                    ylabel="Loss", yscale="log")
        (self.tr_line,) = self.ax.plot([], [], "-o", ms=3, label="Train")
        (self.va_line,) = self.ax.plot([], [], "-o", ms=3, label="Val")
        self.ax.legend(); self.tr_vals, self.va_vals = [], []
    def reset(self):
        self.tr_vals.clear(); self.va_vals.clear()
        self.tr_line.set_data([], []); self.va_line.set_data([], [])
        self.ax.relim(); self.ax.autoscale_view(); self.draw_idle()
    def add_step(self, ep, tr, va):
        self.tr_vals.append(tr); self.va_vals.append(va)
        self.tr_line.set_data(range(1, len(self.tr_vals)+1), self.tr_vals)
        self.va_line.set_data(range(1, len(self.va_vals)+1), self.va_vals)
        self.ax.relim(); self.ax.autoscale_view(); self.draw_idle()

# ═══════════════════ Matplotlib-Canvas – Confusion ===============
class ConfMatCanvas(FigureCanvas):
    clicked = pyqtSignal(int, int)
    def __init__(self, cm: np.ndarray, idx2name: Dict[int, str], parent=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        super().__init__(fig); self.setParent(parent)
        cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
        im = ax.imshow(cm_pct, cmap="Blues")
        cbar = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cbar.ax.set_ylabel("%", rotation=0, labelpad=15, weight="bold")
        ticks = np.arange(len(idx2name))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([idx2name[i] for i in ticks], rotation=45, ha="right")
        ax.set_yticklabels([idx2name[i] for i in ticks])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion-Matrix", fontweight="bold", pad=12)
        thresh = cm_pct.max()/2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0f}%",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] > thresh else "black",
                    fontsize=8, weight="bold")
        fig.tight_layout(); self.ax = ax
        self.mpl_connect("button_press_event", self._on_click)
    def _on_click(self, ev):
        if ev.inaxes != self.ax: return
        self.clicked.emit(int(round(ev.ydata)), int(round(ev.xdata)))

# ═══════════════════ Matplotlib-Canvas – PR-Kurven ===============
class PRCanvas(FigureCanvas):
    def __init__(self, curves: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
                 idx2name: Dict[int, str], parent=None):
        fig, ax = plt.subplots(figsize=(4, 3))
        super().__init__(fig); self.setParent(parent)
        for i, (p, r, au) in curves.items():
            ax.plot(r, p, label=f"{idx2name[i]} AUC={au:.2f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="PR-Kurven")
        ax.legend(); fig.tight_layout()

# ═══════════════════ Training-Thread =============================
class TrainWorker(QThread):
    progress = pyqtSignal(int)
    step     = pyqtSignal(int, float, float)
    finished = pyqtSignal(dict, np.ndarray, dict, float, int, int, object, dict)
    log      = pyqtSignal(str)

    def __init__(self, folder: Path, label_map: Dict[str, int],
                 unknown_id: int, cfg: dict):
        super().__init__()
        self.folder, self.map = folder, label_map
        self.unk, self.cfg = unknown_id, cfg

    def _csvs(self): return [p for p in self.folder.rglob("*.csv")
                             if not p.name.endswith("_track.csv")]

    # --------------------------------------------------------------
    def run(self):                       # noqa: C901
        try:
            seed = self.cfg.get("seed", 42)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            csvs = self._csvs()
            if not csvs:
                raise RuntimeError("Keine CSV-Dateien gefunden.")

            ds_all = IMUWindowDataset(
                csvs, label_map=self.map, window=self.cfg["window"],
                stride=self.cfg["stride"], majority_thr=self.cfg["majority"],
                add_speed=self.cfg["add_speed"], fft_bins=self.cfg["fft_bins"],
                unknown_id=self.unk, augment=False)

            stats = {k: int((ds_all.y == v).sum()) for k, v in self.map.items()}
            self.log.emit("STAT|" + "|".join(f"{k}:{v}" for k, v in stats.items()))

            C_tot = ds_all.X.shape[1]
            uids  = sorted(set(ds_all.y.tolist()))
            id2idx = {u: i for i, u in enumerate(uids)}
            idx2name = {i: next(k for k, v in self.map.items() if v == u)
                        for i, u in enumerate(uids)}

            groups = ds_all.grp
            if self.cfg["cv_folds"] > 1:
                splitter = GroupKFold(n_splits=self.cfg["cv_folds"])
                splits = list(splitter.split(ds_all.X, ds_all.y, groups))
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=self.cfg["split"], random_state=seed)
                splits = list(gss.split(ds_all.X, ds_all.y, groups))
            folds_total = len(splits)
            total_steps = folds_total + (1 if self.cfg.get("final_full") else 0)

            cm_total = np.zeros((len(uids), len(uids)), int)
            rep_acc, curves_fold = None, {}
            fold = 0

            for tr_idx, va_idx in splits:
                if self.isInterruptionRequested():
                    return
                fold += 1
                self.log.emit(f"Fold {fold}/{folds_total} startet …")

                # zuerst GPU, dann evtl. CPU
                devices = []
                if self.cfg.get("use_gpu") and torch.cuda.is_available():
                    devices.append(torch.device("cuda"))
                devices.append(torch.device("cpu"))
                for device in devices:
                    use_amp = device.type == "cuda"
                    try:
                        result = self._train_fold(
                            fold, total_steps, tr_idx, va_idx, ds_all,
                            id2idx, idx2name, C_tot, device, use_amp)
                        if result is None:
                            return
                        cm_fold, rep_fold, curves = result
                        cm_total += cm_fold
                        if rep_acc is None:
                            rep_acc = {k:(v.copy() if isinstance(v, dict) else v)
                                       for k, v in rep_fold.items()}
                        else:
                            for k, v in rep_fold.items():
                                if isinstance(v, dict):
                                    for m in v: rep_acc[k][m] += v[m]
                                else: rep_acc[k] += v
                        curves_fold[fold] = curves
                        break
                    except RuntimeError as e:
                        if ("out of memory" in str(e).lower()
                                and device.type == "cuda"):
                            self.log.emit("⚠️ GPU-OOM → Fallback auf CPU")
                            torch.cuda.empty_cache()
                            continue
                        raise

                self.progress.emit(int(fold/total_steps*100))

            # Durchschnitts-Report
            for k, v in rep_acc.items():
                if isinstance(v, dict):
                    for m in v: rep_acc[k][m] /= folds_total
                else: rep_acc[k] /= folds_total
            τ = _oob_threshold(cm_total.max(1), .95)

            if self.cfg.get("final_full"):
                fold += 1
                self.log.emit("Final training on all data …")
                devices = []
                if self.cfg.get("use_gpu") and torch.cuda.is_available():
                    devices.append(torch.device("cuda"))
                devices.append(torch.device("cpu"))
                for device in devices:
                    use_amp = device.type == "cuda"
                    try:
                        final_model = self._train_all(
                            fold, total_steps, ds_all, id2idx, C_tot,
                            device, use_amp)
                        if final_model is not None:
                            self.last_model_cpu = final_model
                            break
                    except RuntimeError as e:
                        if ("out of memory" in str(e).lower() and device.type == "cuda"):
                            self.log.emit("⚠️ GPU-OOM → Fallback auf CPU")
                            torch.cuda.empty_cache()
                            continue
                        raise

            val_count = len(ds_all)//folds_total if folds_total > 1 else len(va_idx)
            self.finished.emit(rep_acc, cm_total, idx2name, τ,
                               len(ds_all), val_count,
                               self.last_model_cpu, curves_fold)

        except Exception as e:
            self.log.emit(f"Fehler im Training: {e}")

    def _create_model(self, c_in: int, n_cls: int) -> nn.Module:
        arch = self.cfg.get("arch", "HybridNet")
        if arch == "ResNet18":
            return ResNet1D(c_in, n_cls)
        return HybridNet(c_in, n_cls,
                        hidden=self.cfg.get("base", 128),
                        layers=self.cfg.get("layers", 2),
                        drop=self.cfg.get("drop", 0.5))

    # --------------------------------------------------------------
    def _train_fold(self, fold: int, steps_total: int,
                    tr_idx, va_idx, ds_all: IMUWindowDataset,
                    id2idx: Dict[int, int], idx2name: Dict[int, str],
                    C_tot: int, device: torch.device, use_amp: bool):

        X_tr, y_tr = ds_all.X[tr_idx], ds_all.y[tr_idx]
        if self.cfg["augment"]:
            X_tr = np.stack([sign_flip(add_gauss(scale_amplitude(dropout_segment(x))))
                             for x in X_tr])
        y_tr_i = np.array([id2idx[v] for v in y_tr])

        ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
        X_tr_r, y_tr_i_r = ros.fit_resample(X_tr.reshape(len(X_tr), -1), y_tr_i)
        X_tr_r = X_tr_r.reshape(-1, C_tot, self.cfg["window"]).astype(np.float32)

        X_va, y_va = ds_all.X[va_idx], ds_all.y[va_idx]
        y_va_i = np.array([id2idx[v] for v in y_va])

        dl_tr = DataLoader(
            TensorDataset(torch.from_numpy(X_tr_r),
                          torch.from_numpy(y_tr_i_r)),
            batch_size=self.cfg["batch"], shuffle=True,
            pin_memory=(device.type == "cuda"),
        )
        dl_va = DataLoader(
            TensorDataset(torch.from_numpy(X_va),
                          torch.from_numpy(y_va_i)),
            batch_size=self.cfg["batch"], pin_memory=(device.type == "cuda"))

        model = self._create_model(C_tot, len(uids:=id2idx)).to(device)

        alpha = torch.tensor(compute_class_weight(
            "balanced", classes=np.arange(len(uids)), y=y_tr_i_r)).float().to(device)
        crit = FocalLoss(alpha, gamma=1.0,
                         smooth=max(.05, self.cfg["smooth"]))

        opt  = torch.optim.AdamW(model.parameters(),
                                 lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        sched = OneCycleLR(opt, max_lr=self.cfg["lr"],
                           steps_per_epoch=len(dl_tr),
                           epochs=self.cfg["epochs"])
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val, best_state, bad = float("inf"), None, 0
        for ep in range(1, self.cfg["epochs"]+1):
            if self.isInterruptionRequested():
                return None
            model.train(); tr_loss = 0.
            for xb, yb in dl_tr:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb); loss = crit(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(opt); scaler.update(); sched.step()
                tr_loss += loss.item()*len(xb)
            tr_loss /= len(dl_tr.dataset)

            model.eval(); val_loss, probs = 0., []
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        lg = model(xb); loss = crit(lg, yb)
                    val_loss += loss.item()*len(xb)
                    probs.append(torch.softmax(lg, 1).cpu())
            val_loss /= len(dl_va.dataset)
            probs = torch.cat(probs).numpy()

            self.step.emit(ep, tr_loss, val_loss)
            self.progress.emit(int((fold-1+ep/self.cfg["epochs"])/steps_total*100))

            if val_loss < best_val - 1e-4:
                best_val, best_state, bad = val_loss, model.state_dict(), 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break

        model.load_state_dict(best_state)
        model.cpu(); self.last_model_cpu = model

        with torch.no_grad():
            prob_va = torch.softmax(model(torch.from_numpy(X_va)), 1).numpy()
        preds = prob_va.argmax(1)
        cm_fold = confusion_matrix(y_va_i, preds,
                                   labels=range(len(uids)))
        rep_fold = classification_report(
            y_va_i, preds,
            target_names=[idx2name[i] for i in range(len(uids))],
            zero_division=0, output_dict=True)
        curves = pr_curves(y_va_i, prob_va, len(uids))

        return cm_fold, rep_fold, curves

    def _train_all(self, fold: int, steps_total: int,
                   ds_all: IMUWindowDataset, id2idx: Dict[int, int],
                   C_tot: int, device: torch.device, use_amp: bool):
        X_tr = ds_all.X
        if self.cfg["augment"]:
            X_tr = np.stack([
                sign_flip(add_gauss(scale_amplitude(dropout_segment(x))))
                for x in X_tr])
        y_tr_i = np.array([id2idx[v] for v in ds_all.y])

        ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
        X_tr_r, y_tr_i_r = ros.fit_resample(X_tr.reshape(len(X_tr), -1), y_tr_i)
        X_tr_r = X_tr_r.reshape(-1, C_tot, self.cfg["window"]).astype(np.float32)

        dl_tr = DataLoader(
            TensorDataset(torch.from_numpy(X_tr_r),
                          torch.from_numpy(y_tr_i_r)),
            batch_size=self.cfg["batch"], shuffle=True,
            pin_memory=(device.type == "cuda"),
        )

        model = self._create_model(C_tot, len(id2idx)).to(device)
        alpha = torch.tensor(compute_class_weight(
            "balanced", classes=np.arange(len(id2idx)), y=y_tr_i_r)).float().to(device)
        crit = FocalLoss(alpha, gamma=1.0,
                         smooth=max(.05, self.cfg["smooth"]))
        opt  = torch.optim.AdamW(model.parameters(),
                                 lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        sched = OneCycleLR(opt, max_lr=self.cfg["lr"],
                           steps_per_epoch=len(dl_tr),
                           epochs=self.cfg["epochs"])
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for ep in range(1, self.cfg["epochs"]+1):
            if self.isInterruptionRequested():
                return None
            model.train(); tr_loss = 0.
            for xb, yb in dl_tr:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb); loss = crit(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(opt); scaler.update(); sched.step()
                tr_loss += loss.item()*len(xb)
            tr_loss /= len(dl_tr.dataset)

            self.step.emit(ep, tr_loss, tr_loss)
            self.progress.emit(int((fold-1+ep/self.cfg["epochs"]) / steps_total*100))

        model.cpu()
        return model

# ═══════════════════ Qt-GUI ======================================
class TrainingTab(QWidget):
    model_saved = pyqtSignal(str)

    def __init__(self, label_map: Dict[str, int], unknown_id: int,
                 start_folder: Path | None = None):
        super().__init__()
        self.label_map = label_map
        self.unknown_id = unknown_id
        self.folder = start_folder or Path.cwd()
        self.idx2name: dict[int, str] = {}
        self._trained_model: nn.Module | None = None
        self._cfg: dict | None = None
        self._train_info: dict | None = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Road-Anomaly Trainer (HybridNet)")
        pal = self.palette(); pal.setColor(QPalette.Window, QColor("#f7f7f7"))
        self.setPalette(pal)

        splitter = QSplitter(Qt.Horizontal)
        self._build_left(splitter); self._build_right(splitter)

        lay = QVBoxLayout(self); lay.addWidget(splitter)

    # -------- LEFT ------------------------------------------------
    def _build_left(self, splitter):
        left = QWidget(); lv = QVBoxLayout(left)

        row = QHBoxLayout()
        row.addWidget(QLabel("CSV-Ordner:"))
        self.lbl_folder = QLabel(str(self.folder))
        self.lbl_folder.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        row.addWidget(self.lbl_folder, 1)
        btn = QPushButton("Browse…"); btn.clicked.connect(self._browse)
        row.addWidget(btn); lv.addLayout(row)

        self.tbl_stats = QTableWidget()
        self.tbl_stats.setColumnCount(2)
        self.tbl_stats.setHorizontalHeaderLabels(["Class", "Points"])
        self.tbl_stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lv.addWidget(self.tbl_stats, 2)

        form = QFormLayout()
        self.sp_epochs = QSpinBox(minimum=1, maximum=300, value=50)
        self.sp_lr = QDoubleSpinBox(decimals=6, minimum=1e-6,
                                    maximum=1.0, value=5e-4); self.sp_lr.setSingleStep(1e-5)
        self.sp_batch = QSpinBox(minimum=32, maximum=2048, value=256)
        self.sp_wd = QDoubleSpinBox(decimals=6, minimum=0., maximum=.1, value=1e-4)
        self.sp_pat = QSpinBox(minimum=1, maximum=20, value=10)
        self.sp_win = QSpinBox(minimum=64, maximum=1024, value=256)
        self.sp_stride = QSpinBox(minimum=32, maximum=1024, value=128)
        self.sp_major = QDoubleSpinBox(decimals=2, minimum=.5, maximum=1., value=.7)
        self.sp_layers = QSpinBox(minimum=1, maximum=6, value=2)
        self.sp_base = QSpinBox(minimum=16, maximum=512, value=128)
        self.sp_drop = QDoubleSpinBox(decimals=2, minimum=0., maximum=.8, value=.5)
        self.cb_speed = QCheckBox("Speed-Kanal"); self.cb_speed.setChecked(True)
        self.cb_fft = QCheckBox("FFT-Spektrum"); self.cb_fft.setChecked(False)
        self.sp_fft = QSpinBox(minimum=8, maximum=128, value=40)
        self.sp_fft.setEnabled(False); self.cb_fft.toggled.connect(self.sp_fft.setEnabled)
        self.cb_aug = QCheckBox("Augmentation"); self.cb_aug.setChecked(False)
        self.sp_smooth = QDoubleSpinBox(decimals=2, minimum=0., maximum=.2, value=.05)
        self.cb_gpu = QCheckBox("Use GPU"); self.cb_gpu.setChecked(USE_CUDA)
        self.cmb_cv = QComboBox(); self.cmb_cv.addItems(["Holdout", "5-Fold", "K-Fold"])
        self.sp_kfold = QSpinBox(minimum=2, maximum=10, value=5); self.sp_kfold.setEnabled(False)
        self.sp_split = QDoubleSpinBox(decimals=2, minimum=0.1, maximum=0.5, value=0.2)
        self.sp_split.setEnabled(True)
        self.cmb_cv.currentTextChanged.connect(lambda t: self.sp_kfold.setEnabled(t == "K-Fold"))
        self.cmb_cv.currentTextChanged.connect(lambda t: self.sp_split.setEnabled(t == "Holdout"))
        self.sp_seed = QSpinBox(minimum=0, maximum=999999, value=42)
        self.cmb_arch = QComboBox(); self.cmb_arch.addItems(["HybridNet", "ResNet18"])
        self.cb_final = QCheckBox("Final full training"); self.cb_final.setChecked(True)

        widgets = [
            ("Epochs", self.sp_epochs),
            ("Learning Rate", self.sp_lr),
            ("Batch Size", self.sp_batch),
            ("Weight Decay", self.sp_wd),
            ("Patience", self.sp_pat),
            ("Window", self.sp_win),
            ("Stride", self.sp_stride),
            ("Majority ≥", self.sp_major),
            ("LSTM-Layers", self.sp_layers),
            ("Hidden Size", self.sp_base),
            ("Dropout", self.sp_drop),
            ("", self.cb_speed),
            ("", self.cb_fft),
            ("FFT-Bins", self.sp_fft),
            ("Augment.", self.cb_aug),
            ("Label-Smooth", self.sp_smooth),
            ("Use GPU", self.cb_gpu),
            ("Cross-Validation", self.cmb_cv),
            ("K", self.sp_kfold),
            ("Split", self.sp_split),
            ("Seed", self.sp_seed),
            ("Architecture", self.cmb_arch),
            ("Train Full", self.cb_final),
        ]
        for lbl, w in widgets: form.addRow(lbl, w)
        lv.addLayout(form)

        self.btn_train = QPushButton("Train model")
        self.btn_train.clicked.connect(self._start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row_btn = QHBoxLayout(); row_btn.addWidget(self.btn_train); row_btn.addWidget(self.btn_stop)
        lv.addLayout(row_btn)
        self.progress = QProgressBar(); lv.addWidget(self.progress)

        splitter.addWidget(left); splitter.setStretchFactor(0, 1)

    # -------- RIGHT -----------------------------------------------
    def _build_right(self, splitter):
        right = QWidget(); rv = QVBoxLayout(right)
        self.tabs = QTabWidget(); rv.addWidget(self.tabs, 1)

        loss_tab = QWidget(); lt = QVBoxLayout(loss_tab)
        self.loss_canvas = LossCanvas(); lt.addWidget(self.loss_canvas)
        self.tabs.addTab(loss_tab, "Loss")

        self.cm_tab = QWidget(); ct = QVBoxLayout(self.cm_tab)
        self.cm_placeholder = QLabel("Confusion-Matrix\nerscheint nach Training",
                                     alignment=Qt.AlignCenter)
        ct.addWidget(self.cm_placeholder, 1)
        self.tabs.addTab(self.cm_tab, "Confusion")

        self.pr_tab = QWidget(); pt = QVBoxLayout(self.pr_tab)
        self.pr_placeholder = QLabel("PR-Kurven\nerscheinen nach Training",
                                     alignment=Qt.AlignCenter)
        pt.addWidget(self.pr_placeholder, 1)
        self.tabs.addTab(self.pr_tab, "PR-Curves")

        self.met_tab = QWidget(); mt = QVBoxLayout(self.met_tab)
        self.table = QTableWidget(); self.table.setFont(QFont("DejaVu Sans Mono", 9))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        mt.addWidget(self.table, 2)
        self.txt_report = QTextBrowser(); mt.addWidget(self.txt_report, 1)
        self.btn_save = QPushButton("Export Package")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_model)
        self.btn_export = QPushButton("Export ONNX")
        self.btn_export.clicked.connect(self._export_onnx)
        self.btn_load = QPushButton("Load Model")
        self.btn_load.clicked.connect(self._load_model)
        hb = QHBoxLayout(); hb.addWidget(self.btn_save); hb.addWidget(self.btn_export); hb.addWidget(self.btn_load)
        mt.addLayout(hb)
        self.tabs.addTab(self.met_tab, "Metrics")

        splitter.addWidget(right); splitter.setStretchFactor(1, 2)

    # --------------------------------------------------------------
    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "CSV-Ordner wählen", str(self.folder))
        if path: self.folder = Path(path); self.lbl_folder.setText(path)

    def _get_cfg(self) -> dict:
        return dict(
            epochs=self.sp_epochs.value(), lr=self.sp_lr.value(),
            batch=self.sp_batch.value(), wd=self.sp_wd.value(),
            patience=self.sp_pat.value(), window=self.sp_win.value(),
            stride=self.sp_stride.value(), majority=self.sp_major.value(),
            layers=self.sp_layers.value(), base=self.sp_base.value(),
            drop=self.sp_drop.value(), add_speed=self.cb_speed.isChecked(),
            fft_bins=self.sp_fft.value() if self.cb_fft.isChecked() else 0,
            augment=self.cb_aug.isChecked(), smooth=self.sp_smooth.value(),
            use_gpu=self.cb_gpu.isChecked(),
            cv_folds=(0 if self.cmb_cv.currentText()=="Holdout" else
                      (5 if self.cmb_cv.currentText()=="5-Fold" else self.sp_kfold.value())),
            split=self.sp_split.value(),
            seed=self.sp_seed.value(),
            arch=self.cmb_arch.currentText(),
            final_full=self.cb_final.isChecked(),
        )

    # --------------------------------------------------------------
    def _start(self):
        self.btn_train.setEnabled(False); self.progress.reset(); self.loss_canvas.reset()
        if hasattr(self, "cm_scroll"): self.cm_scroll.deleteLater()
        self.cm_placeholder.setText("Training läuft …"); self.cm_placeholder.show()
        if hasattr(self, "pr_scroll"): self.pr_scroll.deleteLater()
        self.pr_placeholder.show(); self.table.clear(); self.txt_report.clear()
        self.btn_save.setEnabled(False)

        cfg = self._get_cfg(); self._cfg = cfg
        self.worker = TrainWorker(self.folder, self.label_map, self.unknown_id, cfg)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.step.connect(self.loss_canvas.add_step)
        self.worker.log.connect(self._log_from_worker)
        self.worker.finished.connect(self._done)
        self.btn_stop.setEnabled(True)
        self.worker.start()

    # --------------------------------------------------------------
    def _log_from_worker(self, msg: str):
        if msg.startswith("STAT|"):
            parts = msg[5:].split("|"); self.tbl_stats.setRowCount(len(parts))
            for r, p in enumerate(parts):
                cls, pts = p.split(":")
                it_cls = QTableWidgetItem(cls); it_pts = QTableWidgetItem(_nice(int(pts)))
                it_cls.setTextAlignment(Qt.AlignCenter); it_pts.setTextAlignment(Qt.AlignCenter)
                self.tbl_stats.setItem(r, 0, it_cls); self.tbl_stats.setItem(r, 1, it_pts)
        else:
            self.txt_report.append(msg)

    # --------------------------------------------------------------
    def _stop(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        self.btn_stop.setEnabled(False)
        self.btn_train.setEnabled(True)

    # --------------------------------------------------------------
    def _done(self, rep_dict, cm, idx2name, τ, n_total, n_val,
              model, curves_fold):
        self.idx2name = idx2name

        # Confusion
        self.cm_placeholder.hide()
        self.cm_scroll = QScrollArea()
        self.cm_scroll.setWidget(ConfMatCanvas(cm, idx2name))
        self.cm_scroll.setWidgetResizable(True)
        self.cm_tab.layout().addWidget(self.cm_scroll, 1)

        # PR-Kurven
        self.pr_placeholder.hide()
        curves_avg: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
        for curves in curves_fold.values():
            for i, cur in curves.items():
                curves_avg.setdefault(i, []).append(cur)
        curves_final = {
            i: (curves_avg[i][0][0], curves_avg[i][0][1],
                np.mean([c[2] for c in curves_avg[i]]))
            for i in curves_avg
        }
        self.pr_scroll = QScrollArea()
        self.pr_scroll.setWidget(PRCanvas(curves_final, idx2name))
        self.pr_scroll.setWidgetResizable(True)
        self.pr_tab.layout().addWidget(self.pr_scroll, 1)

        # Metrics-Tabelle
        classes = list(idx2name.values())
        avg_rows = [
            ("Macro Avg", rep_dict["macro avg"]),
            ("Weighted Avg", rep_dict["weighted avg"]),
            ("Accuracy", {
                "precision": rep_dict["accuracy"],
                "recall": rep_dict["accuracy"],
                "f1-score": rep_dict["accuracy"],
                "support": n_val,
            }),
        ]
        headers = ["Class", "Prec [%]", "Recall [%]", "F1 [%]", "Support"]
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(classes)+len(avg_rows))
        self.table.setHorizontalHeaderLabels(headers)

        for r, cls in enumerate(classes):
            d = rep_dict[cls]; row = [
                cls, f"{d['precision']*100:.1f}", f"{d['recall']*100:.1f}",
                f"{d['f1-score']*100:.1f}", _nice(int(d['support']))]
            for c, v in enumerate(row):
                it = QTableWidgetItem(v); it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, it)

        for i, (name, d) in enumerate(avg_rows, start=len(classes)):
            row = [
                name, f"{d['precision']*100:.1f}", f"{d['recall']*100:.1f}",
                f"{d['f1-score']*100:.1f}", _nice(int(d['support']))]
            for c, v in enumerate(row):
                it = QTableWidgetItem(v); it.setTextAlignment(Qt.AlignCenter)
                it.setBackground(QColor("#e8e8e8"))
                self.table.setItem(i, c, it)

        self.txt_report.append(f"τ = {τ:.3f}   Samples = {_nice(n_total)}")
        self.txt_report.append(
            f"Accuracy: {rep_dict['accuracy']*100:.2f}%   "
            f"Macro-F1: {rep_dict['macro avg']['f1-score']*100:.2f}%   "
            f"Weighted-F1: {rep_dict['weighted avg']['f1-score']*100:.2f}%")

        self._train_info = dict(rep=rep_dict, cm=cm, curves=curves_final, τ=τ,
                                n_total=n_total, n_val=n_val)
        self._trained_model = model
        self.btn_save.setEnabled(True); self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # --------------------------------------------------------------
    def _create_pdf_report(self, pdf_path: Path):
        if not self._train_info or not self._cfg: return
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            ax.set_title("Training-Parameter", fontsize=14, fontweight="bold", pad=20)
            lines = [f"{k}: {v}" for k, v in self._cfg.items()]
            ax.text(0.01, 0.98, "\n".join(lines), va="top", fontsize=10)
            pdf.savefig(fig); plt.close(fig)

            cm = self._train_info["cm"]; idx2name = self.idx2name
            fig, ax = plt.subplots(figsize=(8, 8))
            cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
            im = ax.imshow(cm_pct, cmap="Blues")
            cbar = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
            cbar.ax.set_ylabel("%", rotation=0, labelpad=15, weight="bold")
            ticks = np.arange(len(idx2name))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([idx2name[i] for i in ticks], rotation=45, ha="right")
            ax.set_yticklabels([idx2name[i] for i in ticks])
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title("Confusion Matrix", fontweight="bold", pad=12)
            thresh = cm_pct.max()/2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0f}%",
                        ha="center", va="center",
                        color="white" if cm_pct[i, j] > thresh else "black",
                        fontsize=7, weight="bold")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            curves = self._train_info["curves"]
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, (p, r, au) in curves.items():
                ax.plot(r, p, label=f"{idx2name[i]} AUC={au:.2f}")
            ax.set(xlabel="Recall", ylabel="Precision", title="PR-Kurven")
            ax.legend(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            rep = self._train_info["rep"]
            classes = list(idx2name.values())
            headers = ["Class", "Prec", "Recall", "F1", "Support"]
            rows = [[cls,
                     f"{rep[cls]['precision']*100:.1f}",
                     f"{rep[cls]['recall']*100:.1f}",
                     f"{rep[cls]['f1-score']*100:.1f}",
                     rep[cls]['support']] for cls in classes]
            for name in ["macro avg", "weighted avg"]:
                d = rep[name]
                rows.append([name.replace(" avg", " Avg"),
                             f"{d['precision']*100:.1f}",
                             f"{d['recall']*100:.1f}",
                             f"{d['f1-score']*100:.1f}",
                             d['support']])
            rows.append(["Accuracy", "-", "-",
                         f"{rep['accuracy']*100:.2f}", self._train_info['n_val']])

            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.set_title("Metriken", fontsize=14, fontweight="bold", pad=20)
            table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
            pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.set_title("Zusammenfassung", fontsize=14, fontweight="bold", pad=20)
            lines = [
                f"Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Gesamte Samples: {_nice(self._train_info['n_total'])}",
                f"Validierungs-Samples: {_nice(self._train_info['n_val'])}",
                f"τ-Schwellwert (OOB): {self._train_info['τ']:.3f}",
                f"Accuracy: {rep['accuracy']*100:.2f}%",
                f"Macro-F1: {rep['macro avg']['f1-score']*100:.2f}%",
            ]
            ax.text(0.01, 0.98, "\n".join(lines), va='top', fontsize=11)
            pdf.savefig(fig); plt.close(fig)

    # --------------------------------------------------------------
    def _save_model(self):
        if not self._trained_model or not self._train_info:
            QMessageBox.warning(self, "Kein Modell", "Trainiere zuerst ein Modell!")
            return
        base_dir = QFileDialog.getExistingDirectory(self, "Speicherort wählen", str(Path.home()))
        if not base_dir: return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self._cfg.get('arch','Model')}_{timestamp}"
        folder = Path(base_dir) / model_name; folder.mkdir(parents=True, exist_ok=True)

        pt_path = folder / f"{model_name}.pt"
        torch.save(self._trained_model.state_dict(), pt_path)
        cfg = dict(self._cfg)
        cfg.update({
            "c_in": self._trained_model.in_channels,
            "n_cls": self._trained_model.fc.out_features,
            "arch": self._cfg.get("arch", "HybridNet"),
            "threshold": self._train_info.get("τ") if self._train_info else None,
        })
        with open(folder/"config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        self._create_pdf_report(folder/f"{model_name}.pdf")
        reply = QMessageBox.question(
            self,
            "Export abgeschlossen",
            f"Modell + Bericht gespeichert in:\n{folder}\nIn Hauptprogramm laden?",
        )
        self.txt_report.append(f"Package saved → {folder}")
        if reply == QMessageBox.Yes:
            self.model_saved.emit(str(pt_path))

    # --------------------------------------------------------------
    def _export_onnx(self):
        if not self._trained_model:
            QMessageBox.warning(self, "Kein Modell", "Trainiere zuerst ein Modell!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "ONNX-Export", "", "ONNX (*.onnx)")
        if not path: return
        if not path.endswith(".onnx"): path += ".onnx"
        dummy = torch.randn(1, self._trained_model.in_channels, self.sp_win.value())
        torch.onnx.export(self._trained_model, dummy, path,
                          opset_version=16,
                          input_names=["imu"], output_names=["logits"])
        QMessageBox.information(self, "ONNX-Export",
                                f"Modell erfolgreich exportiert:\n{path}")

    # --------------------------------------------------------------
    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model (*.pt)")
        if not path:
            return
        cfg_path = Path(path).with_name("config.json")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
        else:
            cfg = self._get_cfg()
        self._cfg = cfg
        model = self._create_model(cfg.get("c_in", self.sp_base.value()),
                                   cfg.get("n_cls", len(self.label_map)))
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        self._trained_model = model
        self.txt_report.append(f"Model loaded from {path}")

# ═══════════════════ main() ======================================
if __name__ == "__main__":
    label_map = {
        "normal": 1, "depression": 2, "cover": 3, "cobble": 4,
        "transverse": 5, "gravel": 6, "cracked": 7,
        "bump": 8, "uneven": 9,
    }
    app = QApplication(sys.argv)
    win = TrainingTab(label_map, unknown_id=99, start_folder=Path.home())
    win.resize(1400, 850); win.show()
    sys.exit(app.exec())
