
# -*- coding: utf-8 -*-
"""ISO 2631-1 frequency weighting utilities.

This module implements functions for frequency weighting of vibration
signals according to ISO 2631‑1 and VDI 2057. It uses a lookup table
with weighting factors for the vertical (Wk) and horizontal (Wd)
directions and applies them via FFT multiplication.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import rfft, irfft
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Weighting table
# ---------------------------------------------------------------------------
# Frequency in Hz, factors for Wk (vertical) and Wd (horizontal)
WK_TABLE = np.array([
    (0.10, 0.0312, 0.0624),
    (0.125, 0.0493, 0.0987),
    (0.16, 0.0776, 0.1550),
    (0.20, 0.1210, 0.2420),
    (0.25, 0.1830, 0.3680),
    (0.315, 0.2640, 0.5330),
    (0.40, 0.3500, 0.7100),
    (0.50, 0.4190, 0.8540),
    (0.63, 0.4590, 0.9440),
    (0.80, 0.4770, 0.9910),
    (1.00, 0.4820, 1.0110),
    (1.25, 0.4850, 1.0070),
    (1.60, 0.4930, 0.9710),
    (2.00, 0.5310, 0.8910),
    (2.50, 0.6330, 0.7730),
    (3.15, 0.8070, 0.6400),
    (4.00, 0.9650, 0.5140),
    (5.00, 1.0390, 0.4080),
    (6.30, 1.0540, 0.3230),
    (8.00, 1.0370, 0.2550),
    (10.00, 0.9880, 0.2020),
    (12.5, 0.8990, 0.1600),
    (16.0, 0.7740, 0.1270),
    (20.0, 0.6370, 0.1000),
    (25.0, 0.5100, 0.0796),
    (31.5, 0.4030, 0.0630),
    (40.0, 0.3160, 0.0496),
    (50.0, 0.2450, 0.0387),
    (63.0, 0.1860, 0.0295),
    (80.0, 0.1340, 0.0213),
])

WK_FREQ = WK_TABLE[:, 0]
WK_FACTOR = WK_TABLE[:, 1]
WD_FACTOR = WK_TABLE[:, 2]

WK_INTERP = interp1d(WK_FREQ, WK_FACTOR, bounds_error=False,
                     fill_value=(WK_FACTOR[0], WK_FACTOR[-1]))
WD_INTERP = interp1d(WK_FREQ, WD_FACTOR, bounds_error=False,
                     fill_value=(WD_FACTOR[0], WD_FACTOR[-1]))


def freq_weight_signal_fft(signal: np.ndarray, fs: float,
                           mode: str = "Wk") -> np.ndarray:
    """Apply ISO weighting in the frequency domain.

    Parameters
    ----------
    signal : np.ndarray
        Input time signal.
    fs : float
        Sampling frequency in Hz.
    mode : str
        'Wk' for vertical axis weighting or 'Wd' for horizontal.
    """
    n = len(signal)
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    spec = rfft(signal)
    if mode == "Wk":
        fac = WK_INTERP(freq)
    else:
        fac = WD_INTERP(freq)
    spec *= fac
    return irfft(spec, n)


def exponential_running_rms(x: np.ndarray, fs: float, tau: float = 1.0) -> np.ndarray:
    """Exponential running RMS with time constant *tau* (seconds)."""
    alpha = 1.0 / (tau * fs)
    rms = np.zeros_like(x)
    s2 = 0.0
    for i, val in enumerate(x):
        s2 = (1 - alpha) * s2 + alpha * val * val
        rms[i] = np.sqrt(s2)
    return rms


def calc_awv(ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
             fs: float, comfort: bool = True,
             peak_height: float = 3.19, peak_dist: float = 0.0,
             max_peak: bool = False) -> dict[str, np.ndarray | float]:
    """Calculate weighted vibration values for three axes.

    Returns a dictionary with the weighted signals, running RMS of each
    axis and combined awv as defined in ISO 2631-1.
    """
    awx = freq_weight_signal_fft(ax, fs, "Wd")
    awy = freq_weight_signal_fft(ay, fs, "Wd")
    awz = freq_weight_signal_fft(az, fs, "Wk")

    rms_x = exponential_running_rms(awx, fs)
    rms_y = exponential_running_rms(awy, fs)
    rms_z = exponential_running_rms(awz, fs)

    if comfort:
        kx = ky = kz = 1.0
    else:
        kx = ky = 1.4
        kz = 1.0

    awv = np.sqrt((kx * rms_x) ** 2 + (ky * rms_y) ** 2 + (kz * rms_z) ** 2)
    awv_total = float(np.sqrt(np.mean(awv ** 2)))

    dist_samples = int(peak_dist * fs) if peak_dist > 0 else None
    peaks, _ = find_peaks(awv, height=peak_height,
                          distance=dist_samples)
    if max_peak and len(peaks):
        peaks = np.array([peaks[np.argmax(awv[peaks])]])

    a8 = awv_total * np.sqrt( (8 * 3600) / (len(ax) / fs) )
    crest = float(np.max(np.abs(awv)) / awv_total) if awv_total else float('nan')

    return {
        "awx": awx,
        "awy": awy,
        "awz": awz,
        "rms_x": rms_x,
        "rms_y": rms_y,
        "rms_z": rms_z,
        "awv": awv,
        "awv_total": awv_total,
        "peaks": peaks,
        "A8": a8,
        "crest_factor": crest,
    }
