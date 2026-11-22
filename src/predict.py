"""
predict.py
Load saved models and generate predictions for new observations.
Used by the Streamlit dashboard's Prediction page.
"""

import os
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = "models"


def load_all_models():
    models = {}
    for name, fname in [
        ("Random Forest",  "random_forest.pkl"),
        ("XGBoost",        "xgboost.pkl"),
        ("LightGBM",       "lightgbm.pkl"),
        ("Stacking",       "stacking.pkl"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def predict_single(input_dict: dict, model_name: str = "XGBoost") -> dict:
    """
    Given a dict of feature values and a model name,
    return prediction label + probability.
    """
    models    = load_all_models()
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found.")

    model = models[model_name]

    # Build input dataframe aligned to feature columns
    row = {col: input_dict.get(col, 0) for col in feat_cols}
    X   = pd.DataFrame([row])[feat_cols]

    pred  = int(model.predict(X)[0])
    prob  = float(model.predict_proba(X)[0][1])

    label = "⚠️ Extreme Weather Day" if pred == 1 else "✅ Normal Day"
    return {
        "prediction": pred,
        "probability": round(prob, 4),
        "label": label,
        "model": model_name,
    }


def predict_all_models(input_dict: dict) -> pd.DataFrame:
    """Return predictions from all models for a single input."""
    models    = load_all_models()
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    row       = {col: input_dict.get(col, 0) for col in feat_cols}
    X         = pd.DataFrame([row])[feat_cols]

    rows = []
    for name, model in models.items():
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        rows.append({
            "Model":      name,
            "Prediction": "Extreme" if pred == 1 else "Normal",
            "Probability":f"{prob*100:.1f}%",
        })
    return pd.DataFrame(rows)
