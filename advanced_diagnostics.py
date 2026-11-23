"""V15 advanced vibration diagnostics. Educational tools for synthetic or user-supplied signals."""
from __future__ import annotations
import numpy as np


def signal_features(signal, sample_rate_hz: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 8 or sample_rate_hz <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be finite 1-D data with >=8 samples and positive sample rate")
    rms = float(np.sqrt(np.mean(x*x)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    centered = x - np.mean(x)
    std = float(np.std(centered))
    kurtosis = float(np.mean(centered**4) / std**4) if std > 1e-12 else 0.0
    return {"RMS": rms, "Peak": peak, "Peak-to-Peak": float(np.ptp(x)),
            "Crest Factor": peak / max(rms, 1e-12),
            "Shape Factor": rms / max(mean_abs, 1e-12), "Kurtosis": kurtosis}


def analytic_envelope(signal):
    """Magnitude of the analytic signal using an FFT Hilbert transform."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 8 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be finite 1-D data with >=8 samples")
    n = x.size
    spectrum = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n//2] = 1
        h[1:n//2] = 2
    else:
        h[0] = 1
        h[1:(n+1)//2] = 2
    return np.abs(np.fft.ifft(spectrum*h))


def envelope_spectrum(signal, sample_rate_hz: float):
    """Return one-sided Hann-window envelope spectrum."""
    if sample_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    env = analytic_envelope(signal)
    env = env - np.mean(env)
    win = np.hanning(env.size)
    amp = 2*np.abs(np.fft.rfft(env*win))/max(np.sum(win), 1e-12)
    freq = np.fft.rfftfreq(env.size, d=1/sample_rate_hz)
    if amp.size: amp[0] *= .5
    return freq, amp


def rotational_order_spectrum(signal, sample_rate_hz: float, rpm: float):
    """Map FFT frequency bins to shaft orders; no tachometer resampling is implied."""
    if rpm <= 0:
        raise ValueError("rpm must be positive")
    from signal_diagnostics import calculate_spectrum
    freq, amp = calculate_spectrum(signal, sample_rate_hz)
    return freq/(rpm/60.0), amp


def spectral_band_energy(signal, sample_rate_hz: float, bands=None):
    """Compute normalized FFT energy in user-defined frequency bands."""
    from signal_diagnostics import calculate_spectrum
    bands = bands or {"Low (0-50 Hz)": (0, 50), "Mid (50-200 Hz)": (50, 200), "High (200+ Hz)": (200, sample_rate_hz/2)}
    freq, amp = calculate_spectrum(signal, sample_rate_hz)
    power = amp**2
    total = float(power.sum()) or 1.0
    result = {}
    for name, (lo, hi) in bands.items():
        if lo < 0 or hi <= lo:
            raise ValueError("frequency bands must have 0 <= low < high")
        result[name] = float(power[(freq >= lo) & (freq < hi)].sum()/total)
    return result
