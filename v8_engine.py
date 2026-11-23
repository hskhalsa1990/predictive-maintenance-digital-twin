"""v8 trend analytics and sensor-data quality checks for synthetic PM demos."""
from __future__ import annotations
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["machine_id", "rpm", "load_pct", "temperature_c", "vibration_rms"]


def validate_sensor_frame(frame: pd.DataFrame) -> dict:
    """Return an auditable data-quality summary without modifying the input."""
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    missing_values = int(frame.isna().sum().sum()) if not frame.empty else 0
    duplicate_rows = int(frame.duplicated().sum()) if not frame.empty else 0
    numeric_checks = {}
    for col in ["rpm", "load_pct", "temperature_c", "vibration_rms"]:
        if col in frame:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            numeric_checks[col] = {
                "non_numeric": int(numeric.isna().sum()),
                "negative_values": int((numeric < 0).sum()),
                "finite": bool(np.isfinite(numeric.dropna()).all()),
            }
    return {"rows": int(len(frame)), "missing_columns": missing_columns,
            "missing_values": missing_values, "duplicate_rows": duplicate_rows,
            "numeric_checks": numeric_checks,
            "status": "PASS" if not missing_columns and missing_values == 0 and duplicate_rows == 0 else "REVIEW"}


def trend_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Machine-level first-to-last change summary from accumulated demo batches."""
    if history.empty or "machine_id" not in history:
        return pd.DataFrame(columns=["machine_id", "vibration_delta", "temperature_delta", "trend"])
    records=[]
    for machine, group in history.groupby("machine_id", sort=True):
        ordered=group.sort_values("batch_id") if "batch_id" in group else group
        first, last=ordered.iloc[0], ordered.iloc[-1]
        dv=float(last.get("vibration_rms", 0))-float(first.get("vibration_rms", 0))
        dt=float(last.get("temperature_c", 0))-float(first.get("temperature_c", 0))
        trend="Worsening" if dv > 0.15 or dt > 5 else ("Improving" if dv < -0.15 or dt < -5 else "Stable")
        records.append({"machine_id":machine,"vibration_delta":round(dv,3),"temperature_delta":round(dt,2),"trend":trend})
    return pd.DataFrame(records)


def add_batch_id(frame: pd.DataFrame, batch_id: int) -> pd.DataFrame:
    result=frame.copy()
    result["batch_id"]=int(batch_id)
    return result
