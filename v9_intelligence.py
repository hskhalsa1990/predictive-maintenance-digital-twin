"""v9 fleet intelligence helpers: explainable risk, anomaly detection and synthetic ML.
All outputs are portfolio/demo estimates, not operational alarms or calibrated RUL."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["rpm", "load_pct", "temperature_c", "vibration_rms"]


def health_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a transparent 0-100 health score (100=healthier) and risk band."""
    out = frame.copy()
    v = np.clip((pd.to_numeric(out.vibration_rms, errors="coerce").fillna(0) - 0.2) / 1.8, 0, 1)
    t = np.clip((pd.to_numeric(out.temperature_c, errors="coerce").fillna(25) - 25) / 65, 0, 1)
    load = np.clip(pd.to_numeric(out.load_pct, errors="coerce").fillna(0) / 100, 0, 1)
    risk = 100 * (0.55 * v + 0.35 * t + 0.10 * load)
    out["risk_index"] = np.round(risk, 1)
    out["health_index"] = np.round(100 - risk, 1)
    out["risk_band"] = pd.cut(risk, [-0.01, 25, 50, 75, 100], labels=["Low", "Guarded", "High", "Critical"]).astype(str)
    return out


def detect_iqr_anomalies(frame: pd.DataFrame, columns=FEATURES) -> pd.DataFrame:
    """Flag outliers per sensor using IQR fences; works on a single batch too."""
    out = frame.copy()
    flags = pd.Series(False, index=out.index)
    for col in columns:
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        if pd.notna(spread) and spread > 0:
            flags |= (values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)
    out["iqr_anomaly"] = flags.astype(int)
    return out


def train_synthetic_risk_model(seed: int = 2026, samples: int = 2500):
    """Train a reproducible classifier on generated feature/threshold labels."""
    rng = np.random.default_rng(seed)
    rpm = rng.uniform(700, 3200, samples)
    load = rng.uniform(10, 100, samples)
    temp = 28 + 0.014 * rpm + 0.12 * load + rng.normal(0, 4.0, samples)
    vib = 0.15 + 0.00025 * rpm + 0.005 * load + rng.normal(0, 0.18, samples)
    X = pd.DataFrame({"rpm": rpm, "load_pct": load, "temperature_c": temp, "vibration_rms": np.clip(vib, 0.03, None)})
    # Synthetic target rule is deliberately documented and not a real failure label.
    y = ((X.temperature_c > 78) | (X.vibration_rms > 1.25) | ((X.temperature_c > 70) & (X.vibration_rms > 0.95))).astype(int)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=140, max_depth=8, class_weight="balanced", random_state=seed))
    model.fit(X[FEATURES], y)
    return model


def predict_synthetic_risk(model, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    probabilities = model.predict_proba(out[FEATURES])[:, 1]
    out["synthetic_risk_probability_pct"] = np.round(probabilities * 100, 1)
    out["synthetic_risk_class"] = np.where(probabilities >= 0.5, "Elevated demo risk", "Lower demo risk")
    return out


def project_vibration_crossing(history: pd.DataFrame, threshold: float = 1.25) -> pd.DataFrame:
    """Linear batch projection to threshold, only for machines with >=3 history points."""
    results = []
    if history.empty or not {"machine_id", "batch_id", "vibration_rms"}.issubset(history.columns):
        return pd.DataFrame(columns=["machine_id", "slope_per_batch", "projected_batches_to_threshold", "projection_note"])
    for machine, g in history.groupby("machine_id"):
        g = g.sort_values("batch_id")
        if len(g) < 3:
            results.append({"machine_id": machine, "slope_per_batch": np.nan, "projected_batches_to_threshold": np.nan, "projection_note": "Need 3+ observations"})
            continue
        x = g.batch_id.astype(float).to_numpy(); y = g.vibration_rms.astype(float).to_numpy()
        slope = float(np.polyfit(x, y, 1)[0]); latest = float(y[-1])
        if latest >= threshold:
            estimate = 0.0; note = "At/above illustrative threshold"
        elif slope <= 0:
            estimate = np.nan; note = "No positive trend; projection withheld"
        else:
            estimate = max(0.0, (threshold - latest) / slope); note = "Linear extrapolation; not calibrated RUL"
        results.append({"machine_id": machine, "slope_per_batch": round(slope, 4), "projected_batches_to_threshold": round(estimate, 2) if np.isfinite(estimate) else np.nan, "projection_note": note})
    return pd.DataFrame(results)
