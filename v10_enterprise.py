"""Enterprise demo utilities for predictive maintenance v10. Synthetic-data only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from v9_intelligence import health_index, detect_iqr_anomalies, train_synthetic_risk_model, predict_synthetic_risk
from v8_engine import trend_summary

FEATURES = ["rpm", "load_pct", "temperature_c", "vibration_rms"]


def score_enterprise(frame: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    """Combine transparent score, demo ML probability, anomaly flag, and priority rank."""
    if frame.empty:
        out = frame.copy()
        for col in ["risk_index", "health_index", "risk_band", "iqr_anomaly", "synthetic_risk_probability_pct", "synthetic_risk_class", "enterprise_priority_score", "priority", "recommended_action", "rank"]:
            out[col] = pd.Series(dtype="float64") if col in ["risk_index", "health_index", "iqr_anomaly", "synthetic_risk_probability_pct", "enterprise_priority_score", "rank"] else pd.Series(dtype="object")
        return out
    scored = predict_synthetic_risk(train_synthetic_risk_model(seed), detect_iqr_anomalies(health_index(frame)))
    prob = scored["synthetic_risk_probability_pct"].clip(0, 100)
    health = scored["health_index"].clip(0, 100)
    anomaly = scored["iqr_anomaly"].astype(float) * 100
    scored["enterprise_priority_score"] = np.round(0.50 * (100-health) + 0.35 * prob + 0.15 * anomaly, 1)
    scored["priority"] = pd.cut(scored["enterprise_priority_score"], [-0.01, 25, 50, 75, 100], labels=["Low", "Moderate", "High", "Critical"]).astype(str)
    scored["recommended_action"] = scored["priority"].map({
        "Low": "Continue routine monitoring; verify sensor trends during scheduled checks",
        "Moderate": "Review operating conditions and repeat sensor measurement",
        "High": "Plan focused inspection; verify vibration and temperature with approved instruments",
        "Critical": "Escalate for qualified review; confirm readings before maintenance decision",
    })
    scored["rank"] = scored["enterprise_priority_score"].rank(method="dense", ascending=False).astype(int)
    return scored.sort_values(["enterprise_priority_score", "machine_id"], ascending=[False, True]).reset_index(drop=True)


def make_work_order_queue(scored: pd.DataFrame) -> pd.DataFrame:
    """Draft queue; no automatic dispatch or operational instruction."""
    cols = ["machine_id", "rank", "priority", "enterprise_priority_score", "health_index", "synthetic_risk_probability_pct", "recommended_action"]
    if scored.empty:
        return pd.DataFrame(columns=cols + ["work_order_status"])
    queue = scored.loc[scored.priority.isin(["High", "Critical", "Moderate"]), cols].copy()
    queue["work_order_status"] = "Draft — review required"
    return queue.reset_index(drop=True)



def compare_batches(history: pd.DataFrame) -> pd.DataFrame:
    """Compact machine trend and current status summary."""
    if history.empty:
        return pd.DataFrame(columns=["machine_id", "vibration_delta", "temperature_delta", "trend"])
    return trend_summary(history)
