"""V17 rotor-dynamics visualization and telemetry validation utilities.

Educational engineering models. Outputs are not a substitute for validated rotor analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import welch

REQUIRED_SENSOR_COLUMNS = {"timestamp", "machine_id", "rpm", "temperature_c", "vibration_rms"}

def validate_sensor_csv(frame: pd.DataFrame) -> dict:
    """Validate basic schema, numeric ranges, timestamps and duplicate records."""
    missing = sorted(REQUIRED_SENSOR_COLUMNS - set(frame.columns))
    issues = []
    if missing:
        return {"valid": False, "rows": int(len(frame)), "missing_columns": missing,
                "duplicate_rows": 0, "invalid_numeric_rows": 0, "issues": [f"Missing columns: {', '.join(missing)}"]}
    data = frame.copy()
    numeric = ["rpm", "temperature_c", "vibration_rms"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    bad_num = data[numeric].isna().any(axis=1) | data["timestamp"].isna() | data["machine_id"].isna()
    out_of_range = (data["rpm"] <= 0) | (data["vibration_rms"] < 0)
    duplicates = int(data.duplicated(subset=["timestamp", "machine_id"]).sum())
    if bad_num.any(): issues.append(f"{int(bad_num.sum())} rows contain missing/invalid required values")
    if out_of_range.fillna(False).any(): issues.append(f"{int(out_of_range.fillna(False).sum())} rows have RPM <= 0 or vibration RMS < 0")
    if duplicates: issues.append(f"{duplicates} duplicate machine/timestamp records")
    return {"valid": not issues, "rows": int(len(data)), "missing_columns": [],
            "duplicate_rows": duplicates, "invalid_numeric_rows": int(bad_num.sum()),
            "out_of_range_rows": int(out_of_range.fillna(False).sum()), "issues": issues}

def campbell_data(rpm_values, natural_frequencies_hz=(25.0, 62.0), max_order=4):
    """Return synchronous order lines and fixed natural-frequency lines."""
    rpm = np.asarray(rpm_values, dtype=float)
    if rpm.ndim != 1 or rpm.size < 2 or not np.all(np.isfinite(rpm)) or np.any(rpm <= 0):
        raise ValueError("rpm_values must be a finite positive 1-D array with >=2 values")
    if int(max_order) < 1: raise ValueError("max_order must be >= 1")
    order_lines = {f"{k}X": k * rpm / 60.0 for k in range(1, int(max_order)+1)}
    natural = np.asarray(natural_frequencies_hz, dtype=float)
    if natural.ndim != 1 or not np.all(np.isfinite(natural)) or np.any(natural <= 0):
        raise ValueError("natural frequencies must be finite and positive")
    return rpm, order_lines, natural

def bode_unbalance_response(rpm_values, natural_frequency_hz=30.0, damping_ratio=0.04):
    """Simplified normalized SDOF synchronous-unbalance response and phase."""
    rpm = np.asarray(rpm_values, dtype=float)
    if rpm.ndim != 1 or rpm.size < 2 or not np.all(np.isfinite(rpm)) or np.any(rpm <= 0):
        raise ValueError("rpm_values must be finite positive 1-D values with >=2 points")
    if natural_frequency_hz <= 0 or not 0 < damping_ratio < 1:
        raise ValueError("natural_frequency_hz > 0 and 0 < damping_ratio < 1 required")
    ratio = (rpm / 60.0) / float(natural_frequency_hz)
    den = np.sqrt((1-ratio**2)**2 + (2*damping_ratio*ratio)**2)
    amplitude = ratio**2 / den
    phase = np.degrees(np.arctan2(2*damping_ratio*ratio, 1-ratio**2))
    return rpm, amplitude, phase

def waterfall_spectra(signals, sample_rate_hz, rpm_values, nperseg=1024):
    """Compute Welch spectra for sequential signals and return frequency, RPM, PSD matrix."""
    rows = [np.asarray(s, dtype=float) for s in signals]
    rpm = np.asarray(rpm_values, dtype=float)
    if len(rows) != rpm.size or rpm.size < 2: raise ValueError("signals and rpm_values must have equal length >=2")
    if sample_rate_hz <= 0 or np.any(~np.isfinite(rpm)) or np.any(rpm <= 0): raise ValueError("positive sample rate and RPM required")
    if any(x.ndim != 1 or x.size < 32 or not np.all(np.isfinite(x)) for x in rows): raise ValueError("each signal must be finite 1-D data with >=32 samples")
    n = min(int(nperseg), min(len(x) for x in rows))
    if n < 32: raise ValueError("nperseg must be >=32")
    spectra=[]
    for x in rows:
        f,p = welch(x, fs=sample_rate_hz, nperseg=n, scaling="density")
        spectra.append(p)
    return f, rpm, np.asarray(spectra)
