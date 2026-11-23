"""Synthetic sensor-stream and maintenance-planning helpers for the v7 demo."""
from __future__ import annotations
import numpy as np
import pandas as pd

def generate_sensor_batch(batch_id: int = 0, machines: int = 6, seed: int = 2026) -> pd.DataFrame:
    """Create one reproducible synthetic batch; this is not live equipment telemetry."""
    rng = np.random.default_rng(seed + int(batch_id))
    rows = []
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=5*int(batch_id))
    for i in range(machines):
        drift = max(0, (batch_id - (i+1)*2) * 0.025)
        rpm = float(rng.uniform(900, 3000))
        load = float(rng.uniform(20, 95))
        temp = float(36 + 0.012*rpm + 0.08*load + 10*drift + rng.normal(0, 1.8))
        vib = float(max(0.05, 0.35 + 0.00022*rpm + 0.004*load + drift + rng.normal(0, .12)))
        anomaly = int(vib > 1.35 or temp > 85)
        rows.append({
            "timestamp": timestamp, "machine_id": f"M-{i+1:02d}",
            "rpm": round(rpm, 1), "load_pct": round(load, 1),
            "temperature_c": round(temp, 2), "vibration_rms": round(vib, 3),
            "anomaly_flag": anomaly
        })
    return pd.DataFrame(rows)

def score_maintenance(row) -> tuple[float, str]:
    """Transparent demo score; tune weights and limits with approved engineering data."""
    vib = np.clip(float(row["vibration_rms"]) / 2.0, 0, 1)
    temp = np.clip((float(row["temperature_c"]) - 25.0) / 65.0, 0, 1)
    anomaly = 1.0 if int(row.get("anomaly_flag", 0)) else 0.0
    score = float(100 * (0.50*vib + 0.30*temp + 0.20*anomaly))
    band = "Routine" if score < 30 else ("Plan inspection" if score < 60 else "Priority inspection")
    return round(score, 1), band

def build_work_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a draft maintenance queue from a sensor batch."""
    records = []
    for _, row in frame.iterrows():
        score, band = score_maintenance(row)
        if band != "Routine":
            records.append({
                "machine_id": row["machine_id"], "priority_score": score,
                "priority": band,
                "suggested_action": "Inspect bearing, verify lubrication, and collect confirmatory vibration data",
                "status": "Draft — review required"
            })
    return pd.DataFrame(records, columns=[
        "machine_id", "priority_score", "priority", "suggested_action", "status"
    ])
