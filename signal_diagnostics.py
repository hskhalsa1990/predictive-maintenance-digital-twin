"""Signal-processing utilities for the synthetic predictive-maintenance lab.

Outputs are educational diagnostics, not a certified machinery-diagnostic system.
"""
from __future__ import annotations
import numpy as np


def bearing_characteristic_frequencies(rpm: float, rolling_elements: int = 8,
                                       ball_diameter_mm: float = 10.0,
                                       pitch_diameter_mm: float = 50.0,
                                       contact_angle_deg: float = 0.0) -> dict[str, float]:
    """Return idealized FTF, BSF, BPFO and BPFI in Hz for a rolling bearing."""
    if rpm <= 0 or rolling_elements < 1 or ball_diameter_mm <= 0 or pitch_diameter_mm <= 0:
        raise ValueError("rpm, element count and bearing diameters must be positive")
    if ball_diameter_mm >= pitch_diameter_mm:
        raise ValueError("ball diameter must be smaller than pitch diameter")
    fr = rpm / 60.0
    ratio = (ball_diameter_mm / pitch_diameter_mm) * np.cos(np.deg2rad(contact_angle_deg))
    n = rolling_elements
    return {
        "FTF": float(0.5 * fr * (1 - ratio)),
        "BSF": float((pitch_diameter_mm / (2 * ball_diameter_mm)) * fr * (1 - ratio**2)),
        "BPFO": float(0.5 * n * fr * (1 - ratio)),
        "BPFI": float(0.5 * n * fr * (1 + ratio)),
    }


def generate_vibration_signal(duration_s: float = 4.0, sample_rate_hz: int = 2048,
                              shaft_hz: float = 25.0, fault_hz: float = 90.0,
                              noise_std: float = 0.08, seed: int = 2026):
    """Create reproducible synthetic vibration and its time vector."""
    if duration_s <= 0 or sample_rate_hz < 2 or shaft_hz <= 0 or fault_hz <= 0:
        raise ValueError("duration, sample rate and frequencies must be positive")
    if fault_hz >= sample_rate_hz / 2:
        raise ValueError("fault frequency must be below Nyquist frequency")
    t = np.arange(int(duration_s * sample_rate_hz), dtype=float) / sample_rate_hz
    rng = np.random.default_rng(seed)
    signal = (0.65*np.sin(2*np.pi*shaft_hz*t)
              + 0.22*np.sin(2*np.pi*fault_hz*t)
              + 0.08*np.sin(2*np.pi*2*shaft_hz*t)
              + rng.normal(0, noise_std, size=t.size))
    return t, signal


def calculate_spectrum(signal: np.ndarray, sample_rate_hz: float):
    """Return one-sided Hann-window amplitude spectrum (frequency, amplitude)."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 8 or sample_rate_hz <= 0:
        raise ValueError("signal must be 1-D with at least 8 samples; sample rate must be positive")
    x = x - np.mean(x)
    window = np.hanning(x.size)
    amp = 2.0 * np.abs(np.fft.rfft(x * window)) / max(np.sum(window), 1e-12)
    freq = np.fft.rfftfreq(x.size, d=1.0/sample_rate_hz)
    if amp.size: amp[0] *= 0.5
    return freq, amp


def orbit_coordinates(x_signal: np.ndarray, y_signal: np.ndarray):
    """Pair orthogonal probe signals for a shaft-orbit plot."""
    x, y = np.asarray(x_signal, dtype=float), np.asarray(y_signal, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("x and y signals must be 1-D arrays of equal length >= 2")
    return x, y
