"""Headless training script for the synthetic Predictive Maintenance models.

Generates reproducible synthetic multi-machine data, extracts vibration features,
trains an Isolation Forest (anomaly), a Random Forest classifier (fault type) and a
Random Forest regressor (remaining useful life, RUL), evaluates them on a held-out
machine and writes everything api.py needs to the artifacts folder.

All data is synthetic and the labels are rule-generated. Scores describe how well the
models recover those rules, not real-world failure-prediction performance.

Usage:
    python train_models.py                      # full run, writes artifacts/
    python train_models.py --machines 3 --cycles 40 --out /tmp/pm   # quick run
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, hilbert, sosfilt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 4096
N = 4096
FAULT_CLASSES = ["healthy", "imbalance", "misalignment", "bearing_defect"]


def simulate_temperature(load, ambient_c=25.0, duration_s=900, dt_s=1.0,
                         heat_capacity=18000.0, cooling_coeff=22.0):
    """Lumped first-order energy balance: C*dT/dt = Q - h*(T - Tambient). Illustrative only."""
    steps = int(duration_s / dt_s)
    temp = np.empty(steps, dtype=float)
    temp[0] = ambient_c + 3.0
    q_base = 95.0 + 310.0 * float(load) ** 2
    for k in range(1, steps):
        q = q_base * (1.0 + 0.035 * np.sin(2 * np.pi * k / (steps / 3)))
        temp[k] = temp[k - 1] + (q - cooling_coeff * (temp[k - 1] - ambient_c)) / heat_capacity * dt_s
    return temp


def generate_dataset(machines: int, cycles: int, seed: int):
    """Return (raw frame, list of waveforms). Machine ids are UNIT_1..UNIT_<machines>."""
    if machines < 3 or cycles < 10:
        raise ValueError("need at least 3 machines and 10 cycles (train machines plus one hold-out)")
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    records, waveforms = [], []
    for mi in range(machines):
        machine_bias = rng.normal(0, 0.8)
        for cyc in range(cycles):
            degradation = cyc / (cycles - 1)
            rpm = float(rng.normal(1800 + mi * 9, 22))
            load = float(rng.uniform(0.60, 1.25))
            shaft_hz = rpm / 60.0
            probs = np.array([max(.20, .62 - .40 * degradation), .13, .12, min(.55, .13 + .40 * degradation)])
            fault = str(rng.choice(FAULT_CLASSES, p=probs / probs.sum()))
            thermal = simulate_temperature(load, ambient_c=25.0, duration_s=180, dt_s=1.0)
            temp = float(thermal[-1] + 22 * degradation + machine_bias + rng.normal(0, 0.8))
            amp = (.20 + 1.0 * degradation) * load
            x = amp * np.sin(2 * np.pi * shaft_hz * t)
            x += .16 * amp * np.sin(2 * np.pi * 2 * shaft_hz * t + .3)
            x += (.04 + .15 * degradation) * np.sin(2 * np.pi * 185 * t)
            if fault == "imbalance":
                x += .65 * np.sin(2 * np.pi * shaft_hz * t + .12)
            elif fault == "misalignment":
                x += .62 * np.sin(2 * np.pi * 2 * shaft_hz * t) + .28 * np.sin(2 * np.pi * 3 * shaft_hz * t)
            elif fault == "bearing_defect":
                impact = np.zeros(N)
                impact[np.arange(0, N, max(1, int(FS / (3.6 * shaft_hz))))] = 1.0
                tau = np.arange(180)
                impulse = np.exp(-tau / 28) * np.sin(2 * np.pi * 720 * tau / FS)
                x += .72 * np.convolve(impact, impulse, mode="full")[:N]
            x += rng.normal(0, .09 + .06 * degradation, N)
            records.append(dict(machine_id=f"UNIT_{mi + 1}", cycle=cyc, rpm=rpm, load=load, ambient_c=25.0,
                                temperature_c=temp, degradation=degradation, fault_type=fault))
            waveforms.append(x)
    return pd.DataFrame(records), waveforms


def extract_features(x, rpm, fs=FS):
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    freq = rfftfreq(len(x), 1 / fs)
    spec = np.abs(rfft(x))
    sos = butter(4, [500, 1400], btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfilt(sos, x)))
    env_spec = np.abs(rfft(env - env.mean()))
    out = {"vibration_rms": rms, "vibration_peak": peak, "peak_to_peak": float(np.ptp(x)),
           "crest_factor": float(peak / (rms + 1e-9)), "kurtosis": float(kurtosis(x)), "skewness": float(skew(x)),
           "dominant_frequency_hz": float(freq[1:][np.argmax(spec[1:])]),
           "envelope_rms": float(np.sqrt(np.mean(env ** 2))),
           "envelope_peak_amplitude": float(env_spec[1:][np.argmax(env_spec[1:])])}
    for lo, hi in [(0, 80), (80, 200), (200, 500), (500, 1000), (1000, 2000)]:
        out[f"fft_energy_{lo}_{hi}"] = float(np.sum(spec[(freq >= lo) & (freq < hi)] ** 2))
    for order in (1, 2, 3):
        out[f"order_{order}x_amplitude"] = float(spec[int(np.argmin(np.abs(freq - order * (rpm / 60))))])
    return out


def build_frame(machines: int, cycles: int, seed: int) -> pd.DataFrame:
    raw, waves = generate_dataset(machines, cycles, seed)
    feats = pd.DataFrame([extract_features(w, r) for w, r in zip(waves, raw.rpm)])
    df = pd.concat([raw.reset_index(drop=True), feats], axis=1)
    df["rul_cycles"] = (1 - df.degradation) * (cycles - 1)
    df["dominant_order"] = df.dominant_frequency_hz / (df.rpm / 60.0 + 1e-9)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(("vibration_", "fft_energy_", "order_", "envelope_"))]
    return cols + ["rpm", "load", "temperature_c", "peak_to_peak", "crest_factor",
                   "kurtosis", "skewness", "dominant_frequency_hz", "dominant_order"]


def train(machines: int = 5, cycles: int = 220, seed: int = 42, out_dir: str | Path = "artifacts") -> dict:
    import joblib
    warnings.filterwarnings("ignore")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = build_frame(machines, cycles, seed)
    cols = feature_columns(df)
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    holdout = f"UNIT_{machines}"
    train_mask = df.machine_id != holdout
    test_mask = ~train_mask

    anomaly = make_pipeline(StandardScaler(), IsolationForest(n_estimators=350, contamination=.08, random_state=seed))
    anomaly.fit(X[df.fault_type == "healthy"])
    fault = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=350, class_weight="balanced", min_samples_leaf=2, random_state=seed))
    fault.fit(X[train_mask], df.loc[train_mask, "fault_type"])
    rul = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    rul.fit(X[train_mask], df.loc[train_mask, "rul_cycles"])

    pred = fault.predict(X[test_mask]); rul_pred = np.maximum(0, rul.predict(X[test_mask]))
    y_f, y_r = df.loc[test_mask, "fault_type"], df.loc[test_mask, "rul_cycles"]
    metrics = {
        "holdout_machine": holdout, "holdout_rows": int(test_mask.sum()),
        "fault_accuracy": round(float(accuracy_score(y_f, pred)), 4),
        "fault_macro_f1": round(float(f1_score(y_f, pred, average="macro", zero_division=0)), 4),
        "rul_mae_cycles": round(float(mean_absolute_error(y_r, rul_pred)), 3),
        "rul_rmse_cycles": round(float(mean_squared_error(y_r, rul_pred) ** .5), 3),
        "rul_r2": round(float(r2_score(y_r, rul_pred)), 4),
        "notice": "Synthetic data with rule-generated labels; not evidence of real-world performance.",
    }
    df.to_csv(out / "synthetic_sensor_dataset.csv", index=False)
    joblib.dump(anomaly, out / "anomaly_model.joblib")
    joblib.dump(fault, out / "fault_classifier.joblib")
    joblib.dump(rul, out / "rul_regressor.joblib")
    (out / "feature_columns.json").write_text(json.dumps(cols, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "metadata.json").write_text(json.dumps({
        "project": "Predictive Maintenance 4.0", "synthetic_data": True, "seed": seed,
        "sampling_frequency_hz": FS, "samples_per_snapshot": N, "machines": machines, "cycles": cycles,
        "train_machines": [f"UNIT_{i}" for i in range(1, machines)], "holdout_machine": holdout,
        "fault_classes": FAULT_CLASSES,
        "thermal_model": "illustrative lumped first-order energy balance"}, indent=2))
    return metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--machines", type=int, default=5)
    p.add_argument("--cycles", type=int, default=220)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts")
    a = p.parse_args(argv)
    m = train(a.machines, a.cycles, a.seed, a.out)
    print(json.dumps(m, indent=2))
    print(f"Artifacts written to {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
