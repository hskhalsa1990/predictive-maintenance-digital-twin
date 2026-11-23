"""V16 rotor-dynamics tools: band-limited envelope, tachometer order tracking and bearing peak checks.

Educational signal-processing utilities. Requires scipy; not a certified diagnostic system.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks


def bandpass_envelope_spectrum(signal, sample_rate_hz: float, low_hz: float, high_hz: float):
    """Band-pass a vibration signal, compute its Hilbert envelope and one-sided spectrum."""
    x = np.asarray(signal, dtype=float)
    nyq = sample_rate_hz / 2.0
    if x.ndim != 1 or x.size < 32 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be finite 1-D data with >=32 samples")
    if sample_rate_hz <= 0 or not (0 < low_hz < high_hz < nyq):
        raise ValueError("require 0 < low < high < Nyquist and positive sample rate")
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, x)
    env = np.abs(hilbert(filtered))
    env = env - np.mean(env)
    win = np.hanning(x.size)
    amp = 2 * np.abs(np.fft.rfft(env * win)) / max(np.sum(win), 1e-12)
    freq = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    if amp.size:
        amp[0] *= 0.5
    return filtered, env, freq, amp


def tachometer_order_track(signal, sample_rate_hz: float, tach_pulse_times_s, samples_per_rev: int = 256):
    """Angularly resample signal using one tach pulse per revolution, then return order spectrum."""
    x = np.asarray(signal, dtype=float)
    pulses = np.asarray(tach_pulse_times_s, dtype=float)
    if x.ndim != 1 or x.size < 64 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be finite 1-D data with >=64 samples")
    if sample_rate_hz <= 0 or samples_per_rev < 32:
        raise ValueError("positive sample rate and >=32 samples/rev required")
    if pulses.ndim != 1 or pulses.size < 3 or not np.all(np.isfinite(pulses)) or np.any(np.diff(pulses) <= 0):
        raise ValueError("tach pulses must be >=3 finite, strictly increasing times")
    duration = x.size / sample_rate_hz  # acquisition window duration (exclusive final sample time)
    if pulses[0] < 0 or pulses[-1] > duration:
        raise ValueError("tach pulse times must lie within signal duration")
    pulse_idx = pulses * sample_rate_hz
    rev_grid = np.arange(pulses.size)
    sample_grid = np.arange(x.size)
    revolution_at_sample = np.interp(sample_grid, pulse_idx, rev_grid, left=np.nan, right=np.nan)
    valid = np.isfinite(revolution_at_sample)
    theta = revolution_at_sample[valid]
    start_rev, end_rev = float(theta[0]), float(theta[-1])
    uniform_rev = np.arange(start_rev, end_rev, 1.0 / samples_per_rev)
    if uniform_rev.size < samples_per_rev * 2:
        raise ValueError("tachometer record must cover at least two complete revolutions")
    angular_signal = np.interp(uniform_rev, theta, x[valid])
    angular_signal -= np.mean(angular_signal)
    win = np.hanning(angular_signal.size)
    amp = 2 * np.abs(np.fft.rfft(angular_signal * win)) / max(np.sum(win), 1e-12)
    orders = np.fft.rfftfreq(angular_signal.size, d=1.0 / samples_per_rev)
    if amp.size:
        amp[0] *= 0.5
    return uniform_rev, angular_signal, orders, amp


def bearing_peak_match(order_or_hz, amplitude, expected_frequency, tolerance: float = 0.08):
    """Find the strongest local spectral peak near each ideal bearing frequency."""
    f = np.asarray(order_or_hz, dtype=float)
    a = np.asarray(amplitude, dtype=float)
    if f.ndim != 1 or a.ndim != 1 or f.size != a.size or f.size < 3:
        raise ValueError("frequency and amplitude arrays must be equal 1-D arrays")
    if tolerance <= 0 or not np.all(np.isfinite(f)) or not np.all(np.isfinite(a)):
        raise ValueError("finite spectrum and positive tolerance required")
    peaks, _ = find_peaks(a)
    result = []
    for label, target in expected_frequency.items():
        if target <= 0:
            raise ValueError("expected bearing frequencies must be positive")
        candidates = peaks[np.abs(f[peaks] - target) <= target * tolerance]
        if candidates.size:
            idx = int(candidates[np.argmax(a[candidates])])
            result.append({"fault": label, "expected_frequency": float(target), "matched_peak": float(f[idx]),
                           "amplitude": float(a[idx]), "relative_error_pct": float(100 * abs(f[idx]-target)/target), "peak_found": True})
        else:
            result.append({"fault": label, "expected_frequency": float(target), "matched_peak": np.nan,
                           "amplitude": 0.0, "relative_error_pct": np.nan, "peak_found": False})
    return result
