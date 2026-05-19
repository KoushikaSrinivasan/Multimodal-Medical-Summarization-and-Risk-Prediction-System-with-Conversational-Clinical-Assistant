"""
Train XGBoost readmission risk model on the Diabetes 130-US Hospitals dataset.

Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
Download diabetic_data.csv and place it in data/ before running this script.

Usage:
    python train/train_risk_model.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    import shap
except ImportError:
    print("Install dependencies: pip install xgboost scikit-learn shap pandas")
    sys.exit(1)

DATA_PATH = Path("data/diabetic_data.csv")
MODEL_OUTPUT = Path("train/risk_model.json")


def load_and_preprocess(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path, na_values=["?"])

    # Binary readmission label: readmitted within 30 days = 1
    df["label"] = (df["readmitted"] == "<30").astype(int)

    # Age band → numeric midpoint
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age_num"] = df["age"].map(age_map).fillna(65)

    # Select features that map to risk_predictor.py's feature_order
    df["has_diabetes"] = df["diag_1"].str.startswith("250", na=False).astype(int)
    df["has_heart_disease"] = df["diag_1"].isin(
        ["410", "411", "412", "428"] + [str(i) for i in range(410, 415)]
    ).astype(int)
    df["has_copd"] = df["diag_1"].isin([str(i) for i in range(490, 497)]).astype(int)
    df["has_ckd"] = df["diag_1"].isin(["585", "586"]).astype(int)
    df["has_high_risk_condition"] = (
        df["has_diabetes"] | df["has_heart_disease"] | df["has_copd"] | df["has_ckd"]
    ).astype(int)

    feature_cols = [
        "age_num",               # → age
        "number_emergency",      # → num_prior_admissions (proxy)
        "number_diagnoses",      # → num_conditions
        "num_medications",       # → num_medications
        "has_high_risk_condition",
        "has_diabetes",
        "has_heart_disease",
        "has_copd",
        "has_ckd",
    ]

    # has_abnormal_labs: A1Cresult or max_glu_serum not normal
    df["has_abnormal_labs"] = (
        (df.get("A1Cresult", ">7") != "None") |
        (df.get("max_glu_serum", ">200") != "None")
    ).astype(int)
    feature_cols.append("has_abnormal_labs")

    df["time_in_hospital"] = df["time_in_hospital"].fillna(3)
    feature_cols.append("time_in_hospital")

    df = df.dropna(subset=feature_cols + ["label"])
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    final_feature_names = [
        "age", "num_prior_admissions", "num_conditions", "num_medications",
        "has_high_risk_condition", "has_diabetes", "has_heart_disease",
        "has_copd", "has_ckd", "has_abnormal_labs", "time_in_hospital",
    ]
    return X, y, final_feature_names


def train(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> xgb.Booster:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),  # handle class imbalance
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=50,
    )

    # Evaluation
    preds = model.predict(dval)
    auc = roc_auc_score(y_val, preds)
    print(f"\nValidation AUC: {auc:.4f}")
    print(classification_report(y_val, (preds >= 0.5).astype(int)))

    return model


def generate_shap_summary(model: xgb.Booster, X: np.ndarray, feature_names: list[str]):
    print("\nGenerating SHAP summary...")
    sample = X[:1000]
    dmatrix = xgb.DMatrix(sample, feature_names=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dmatrix)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    for feat, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1]):
        print(f"  {feat:35s}: {val:.4f}")


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        print("Download from: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008")
        print("Place diabetic_data.csv in the data/ folder and re-run.")
        return

    print("Loading dataset...")
    X, y, feature_names = load_and_preprocess(DATA_PATH)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive (readmitted <30d): {y.sum()} ({y.mean():.1%})")

    print("\nTraining XGBoost model...")
    model = train(X, y, feature_names)

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUTPUT))
    print(f"\nModel saved to {MODEL_OUTPUT}")

    generate_shap_summary(model, X, feature_names)


if __name__ == "__main__":
    main()
