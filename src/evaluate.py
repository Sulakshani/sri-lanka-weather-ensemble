"""
evaluate.py — evaluate all 5 models on held-out test set.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    models = {}
    for name, fname in [
        ("Random Forest", "random_forest.pkl"),
        ("XGBoost",       "xgboost.pkl"),
        ("LightGBM",      "lightgbm.pkl"),
        ("CatBoost",      "catboost.pkl"),
        ("Stacking",      "stacking.pkl"),
    ]:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)
    return models


def compute_metrics(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
        "CM":        confusion_matrix(y_test, y_pred).tolist(),
    }


def baseline_metrics(y_test):
    y_pred = np.zeros(len(y_test), dtype=int)
    return {
        "Model": "Majority Baseline",
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0, "ROC-AUC": 0.5,
        "CM": confusion_matrix(y_test, y_pred).tolist(),
    }


def feature_importance_df(model, feat_cols, model_name):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature":    feat_cols,
            "importance": model.feature_importances_,
            "model":      model_name,
        }).sort_values("importance", ascending=False).head(20)
    return None


def evaluate_all(data):
    models    = load_models()
    X_test    = data["X_test"].values
    y_test    = data["y_test"]
    feat_cols = data["feat_cols"]

    results   = [baseline_metrics(y_test.values)]
    fi_frames = []

    for name, model in models.items():
        print(f"Evaluating {name} …")
        m  = compute_metrics(name, model, X_test, y_test)
        results.append(m)
        fi = feature_importance_df(model, feat_cols, name)
        if fi is not None:
            fi_frames.append(fi)

    df_results = pd.DataFrame(results).drop(columns=["CM"])
    df_results.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    print("\n=== Model Comparison (Test Set) ===")
    print(df_results.to_string(index=False))

    with open(os.path.join(OUTPUT_DIR, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if fi_frames:
        pd.concat(fi_frames).to_csv(
            os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    best = (df_results[df_results["Model"] != "Majority Baseline"]
            .sort_values("F1 Score", ascending=False).iloc[0])
    print(f"\n🏆 Best: {best['Model']}  F1={best['F1 Score']}  ROC-AUC={best['ROC-AUC']}")
    return results, df_results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import run_preprocessing
    data = run_preprocessing()
    evaluate_all(data)
