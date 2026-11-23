"""v11 reproducible synthetic model validation and explainability."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

FEATURES = ["rpm", "load_pct", "temperature_c", "vibration_rms"]

def make_synthetic_validation_data(seed: int = 2026, samples: int = 3000):
    rng = np.random.default_rng(seed)
    rpm = rng.uniform(700, 3200, samples)
    load = rng.uniform(10, 100, samples)
    temp = 28 + .014*rpm + .12*load + rng.normal(0, 4, samples)
    vib = .15 + .00025*rpm + .005*load + rng.normal(0, .18, samples)
    frame = pd.DataFrame({"rpm": rpm, "load_pct": load, "temperature_c": temp,
                        "vibration_rms": np.clip(vib, .03, None)})
    labels = ((frame.temperature_c > 78) | (frame.vibration_rms > 1.25) |
              ((frame.temperature_c > 70) & (frame.vibration_rms > .95))).astype(int)
    return frame, labels

def evaluate_synthetic_model(seed: int = 2026):
    X, y = make_synthetic_validation_data(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=160, max_depth=9, min_samples_leaf=2,
        class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test); prob = model.predict_proba(X_test)[:, 1]
    metrics = {"Accuracy": accuracy_score(y_test, pred),
              "Precision": precision_score(y_test, pred, zero_division=0),
              "Recall": recall_score(y_test, pred, zero_division=0),
              "F1": f1_score(y_test, pred, zero_division=0),
              "ROC-AUC": roc_auc_score(y_test, prob)}
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    importance = permutation_importance(model, X_test, y_test, n_repeats=5,
        random_state=seed, scoring="f1", n_jobs=-1)
    feature = pd.DataFrame({"feature": FEATURES,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    return model, metrics, cm, feature, X_test, y_test, prob
