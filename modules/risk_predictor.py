"""
Readmission and emergency risk prediction using XGBoost.
Falls back to a rule-based scorer if the trained model is not yet available.
"""

import json
import os
import numpy as np
from pathlib import Path
from config import RISK_MODEL_PATH


try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


_model: "xgb.Booster | None" = None


def _get_model() -> "xgb.Booster | None":
    global _model
    if _model is not None:
        return _model
    if not _XGB_AVAILABLE:
        return None
    if Path(RISK_MODEL_PATH).exists():
        _model = xgb.Booster()
        _model.load_model(RISK_MODEL_PATH)
    return _model


def predict_risk(entities: dict, patient_meta: dict | None = None) -> dict:
    """
    Predict readmission risk and emergency risk from clinical entities.

    patient_meta keys (all optional):
        age, num_prior_admissions, has_diabetes, has_heart_disease,
        has_copd, has_ckd, num_medications, time_in_hospital

    Returns:
        {
          "readmission_risk": 0.72,
          "emergency_risk": 0.45,
          "severity": "HIGH",
          "features": {...},
        }
    """
    features = _build_features(entities, patient_meta or {})
    model = _get_model()

    if model is not None and _XGB_AVAILABLE:
        readmission_risk = _xgb_predict(model, features)
    else:
        readmission_risk = _rule_based_score(features)

    emergency_risk = _estimate_emergency_risk(features, readmission_risk)

    severity = _classify_severity(readmission_risk)

    return {
        "readmission_risk": round(readmission_risk, 3),
        "emergency_risk": round(emergency_risk, 3),
        "severity": severity,
        "features": features,
    }


def _build_features(entities: dict, meta: dict) -> dict:
    high_risk_conditions = {
        "diabetes", "heart failure", "copd", "ckd", "chronic kidney disease",
        "pneumonia", "sepsis", "myocardial infarction", "atrial fibrillation",
        "hypertension", "cancer", "cirrhosis", "renal failure",
    }
    diseases_lower = {d.lower() for d in entities.get("diseases", [])}

    has_high_risk = any(cond in d for cond in high_risk_conditions for d in diseases_lower)
    num_conditions = len(entities.get("diseases", []))
    num_medications = len(entities.get("medications", []))
    has_abnormal_labs = len(entities.get("labs", [])) > 3

    return {
        "age": int(meta.get("age", 65)),
        "num_prior_admissions": int(meta.get("num_prior_admissions", 1)),
        "num_conditions": num_conditions,
        "num_medications": num_medications,
        "has_high_risk_condition": int(has_high_risk),
        "has_diabetes": int("diabetes" in diseases_lower or meta.get("has_diabetes", False)),
        "has_heart_disease": int(
            any(k in diseases_lower for k in ["heart failure", "myocardial infarction", "cardiac"])
            or meta.get("has_heart_disease", False)
        ),
        "has_copd": int("copd" in diseases_lower or meta.get("has_copd", False)),
        "has_ckd": int(
            any(k in diseases_lower for k in ["ckd", "chronic kidney", "renal failure"])
            or meta.get("has_ckd", False)
        ),
        "has_abnormal_labs": int(has_abnormal_labs),
        "time_in_hospital": int(meta.get("time_in_hospital", 3)),
    }


def _xgb_predict(model: "xgb.Booster", features: dict) -> float:
    import xgboost as xgb
    feature_order = [
        "age", "num_prior_admissions", "num_conditions", "num_medications",
        "has_high_risk_condition", "has_diabetes", "has_heart_disease",
        "has_copd", "has_ckd", "has_abnormal_labs", "time_in_hospital",
    ]
    x = np.array([[features[k] for k in feature_order]], dtype=np.float32)
    dmatrix = xgb.DMatrix(x, feature_names=feature_order)
    prob = model.predict(dmatrix)[0]
    return float(np.clip(prob, 0.0, 1.0))


def _rule_based_score(features: dict) -> float:
    """Heuristic fallback when trained model is absent."""
    score = 0.0
    score += min(features["age"] / 100, 0.3)
    score += features["num_prior_admissions"] * 0.08
    score += features["num_conditions"] * 0.04
    score += features["num_medications"] * 0.02
    score += features["has_high_risk_condition"] * 0.20
    score += features["has_diabetes"] * 0.06
    score += features["has_heart_disease"] * 0.10
    score += features["has_copd"] * 0.06
    score += features["has_ckd"] * 0.08
    score += features["has_abnormal_labs"] * 0.05
    score += min(features["time_in_hospital"] / 30, 0.10)
    return float(np.clip(score, 0.0, 1.0))


def _estimate_emergency_risk(features: dict, readmission_risk: float) -> float:
    base = readmission_risk * 0.6
    if features["has_heart_disease"]:
        base += 0.15
    if features["has_ckd"]:
        base += 0.10
    if features["has_abnormal_labs"]:
        base += 0.05
    return float(np.clip(base, 0.0, 1.0))


def _classify_severity(score: float) -> str:
    if score >= 0.7:
        return "CRITICAL"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"
