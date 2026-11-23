from pathlib import Path
import json, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
app=FastAPI(title="Predictive Maintenance 4.0 API",version="4.0.0")

class SensorFeatures(BaseModel):
    rpm: float
    load: float
    temperature_c: float
    vibration_rms: float
    vibration_peak: float
    peak_to_peak: float
    crest_factor: float
    kurtosis: float
    skewness: float
    dominant_frequency_hz: float
    envelope_rms: float
    envelope_peak_amplitude: float
    fft_energy_0_80: float
    fft_energy_80_200: float
    fft_energy_200_500: float
    fft_energy_500_1000: float
    fft_energy_1000_2000: float
    order_1x_amplitude: float
    order_2x_amplitude: float
    order_3x_amplitude: float
    dominant_order: float

@app.get("/")
def root():
    return {"project":"Predictive Maintenance 4.0","docs":"/docs"}

@app.post("/predict")
def predict(payload: SensorFeatures):
    try:
        cols=json.loads((ART/"feature_columns.json").read_text())
        row=pd.DataFrame([[getattr(payload,c) for c in cols]],columns=cols)
        anomaly=joblib.load(ART/"anomaly_model.joblib")
        fault=joblib.load(ART/"fault_classifier.joblib")
        rul=joblib.load(ART/"rul_regressor.joblib")
        return {
            "potential_anomaly":bool(anomaly.predict(row)[0] == -1),
            "predicted_fault":str(fault.predict(row)[0]),
            "estimated_rul_cycles":round(max(0,float(rul.predict(row)[0])),2),
            "notice":"Synthetic portfolio model; not approved for operational use."
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503,detail="Run the notebook to generate model artifacts.")
