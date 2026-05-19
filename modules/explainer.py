"""
SHAP-based explainability for the XGBoost risk prediction model.
Generates feature importance explanations for readmission risk scores.
"""

import numpy as np
from typing import Any


try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


FEATURE_LABELS = {
    "age": "Patient Age",
    "num_prior_admissions": "Prior Hospital Admissions",
    "num_conditions": "Number of Diagnoses",
    "num_medications": "Number of Medications",
    "has_high_risk_condition": "High-Risk Condition Present",
    "has_diabetes": "Diabetes",
    "has_heart_disease": "Heart Disease",
    "has_copd": "COPD / Emphysema",
    "has_ckd": "Chronic Kidney Disease",
    "has_abnormal_labs": "Abnormal Lab Values",
    "time_in_hospital": "Days in Hospital",
}

FEATURE_ORDER = list(FEATURE_LABELS.keys())


def explain_risk(model: Any, features: dict) -> dict:
    """
    Generate SHAP explanation for a single patient's risk prediction.

    Returns:
        {
          "shap_values": {feature: shap_value, ...},
          "top_factors": [{"feature": ..., "label": ..., "impact": ..., "direction": ...}],
          "explanation_text": "...",
          "base_value": float,
        }
    """
    if not _SHAP_AVAILABLE or not _XGB_AVAILABLE or model is None:
        return _rule_based_explanation(features)

    x = np.array([[features[k] for k in FEATURE_ORDER]], dtype=np.float32)
    dmatrix = xgb.DMatrix(x, feature_names=FEATURE_ORDER)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(dmatrix)

    shap_dict = {k: float(shap_vals[0][i]) for i, k in enumerate(FEATURE_ORDER)}
    base_value = float(explainer.expected_value)

    top_factors = _build_top_factors(shap_dict, features)
    explanation_text = _build_explanation_text(top_factors, features)

    return {
        "shap_values": shap_dict,
        "top_factors": top_factors,
        "explanation_text": explanation_text,
        "base_value": base_value,
    }


def _build_top_factors(shap_dict: dict, features: dict) -> list[dict]:
    factors = []
    for feature, shap_val in shap_dict.items():
        if abs(shap_val) < 0.001:
            continue
        factors.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": features.get(feature),
            "shap_value": round(shap_val, 4),
            "impact": abs(shap_val),
            "direction": "increases" if shap_val > 0 else "decreases",
        })

    factors.sort(key=lambda x: x["impact"], reverse=True)
    return factors[:6]


def _build_explanation_text(top_factors: list[dict], features: dict) -> str:
    if not top_factors:
        return "Insufficient data to generate explanation."

    lines = ["The readmission risk is driven primarily by:"]
    for i, f in enumerate(top_factors[:4], 1):
        lines.append(
            f"  {i}. {f['label']} (value: {f['value']}) — {f['direction']} risk "
            f"[SHAP: {f['shap_value']:+.3f}]"
        )

    return "\n".join(lines)


def _rule_based_explanation(features: dict) -> dict:
    """Fallback when SHAP or XGBoost is not available."""
    weights = {
        "has_heart_disease": 0.10,
        "has_ckd": 0.08,
        "has_high_risk_condition": 0.20,
        "num_prior_admissions": 0.08,
        "has_diabetes": 0.06,
        "has_copd": 0.06,
        "has_abnormal_labs": 0.05,
        "num_conditions": 0.04,
        "num_medications": 0.02,
        "age": 0.003,
        "time_in_hospital": 0.003,
    }

    shap_dict = {k: weights.get(k, 0) * features.get(k, 0) for k in FEATURE_ORDER}
    top_factors = _build_top_factors(shap_dict, features)
    explanation_text = _build_explanation_text(top_factors, features)

    return {
        "shap_values": shap_dict,
        "top_factors": top_factors,
        "explanation_text": explanation_text,
        "base_value": 0.3,
    }
