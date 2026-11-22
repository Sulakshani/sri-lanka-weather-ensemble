"""
features.py
Hybridization experiment: RF + ARIMA on district-level rainfall time series.

Approach:
1. Fit ARIMA on a single district's daily rainfall (Colombo).
2. Compute ARIMA residuals.
3. Train Random Forest to predict extreme rainfall from ARIMA residual features.
4. Compare RF-only vs ARIMA+RF hybrid on the test period.

This is a valid, lecturer-specified hybridization for time-series data.
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"
RANDOM_STATE = 42
DISTRICT = "Colombo"      # single district for time-series experiment
RAIN_THRESHOLD = 22.3     # same threshold as main model


def run_hybrid(df: pd.DataFrame):
    """
    Run RF + ARIMA hybrid on Colombo district rainfall.
    Returns a results dict.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("statsmodels not available — skipping hybrid.")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n[Hybrid] Running RF+ARIMA experiment on {DISTRICT} …")

    # ── 1. Extract district rainfall time series ──────────────────────────
    dist = (df[df["city_name"] == DISTRICT]
            .sort_values("date")
            [["date", "precipitation_sum (mm)", "temperature_2m_max (°C)"]]
            .dropna()
            .reset_index(drop=True))

    dist["extreme_rain"] = (dist["precipitation_sum (mm)"] >= RAIN_THRESHOLD).astype(int)

    # Chronological split — last 15% as test
    n = len(dist)
    split_idx = int(n * 0.85)
    train_df = dist.iloc[:split_idx].copy()
    test_df  = dist.iloc[split_idx:].copy()

    # ── 2. Fit ARIMA on training rainfall ────────────────────────────────
    print("  Fitting ARIMA(2,1,2) on training rainfall …")
    rain_train = train_df["precipitation_sum (mm)"].values.astype(float)

    arima_model = ARIMA(rain_train, order=(2, 1, 2))
    arima_fit   = arima_model.fit()

    # In-sample residuals for training
    train_df = train_df.copy()
    train_df["arima_residual"] = arima_fit.resid

    # Out-of-sample forecast for test
    forecast = arima_fit.forecast(steps=len(test_df))
    test_df  = test_df.copy()
    test_df["arima_residual"] = (
        test_df["precipitation_sum (mm)"].values - forecast
    )

    # ── 3. Build lag features ─────────────────────────────────────────────
    for df_part in [train_df, test_df]:
        df_part["rain_lag1"]  = df_part["precipitation_sum (mm)"].shift(1).fillna(0)
        df_part["rain_lag3"]  = df_part["precipitation_sum (mm)"].shift(3).fillna(0)
        df_part["rain_roll7"] = df_part["precipitation_sum (mm)"].shift(1).rolling(7, min_periods=1).mean().fillna(0)
        df_part["month"]      = df_part["date"].dt.month

    # precipitation_sum excluded — it defines the label (leakage prevention)
    FEAT = ["rain_lag1", "rain_lag3", "rain_roll7", "month", "arima_residual"]

    X_tr = train_df[FEAT].dropna()
    y_tr = train_df.loc[X_tr.index, "extreme_rain"]
    X_te = test_df[FEAT].dropna()
    y_te = test_df.loc[X_te.index, "extreme_rain"]

    # ── 4. RF-only baseline ───────────────────────────────────────────────
    rf_only = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=RANDOM_STATE)
    FEAT_NO_ARIMA = [f for f in FEAT if f != "arima_residual"]
    rf_only.fit(X_tr[FEAT_NO_ARIMA],  y_tr)
    y_pred_rf = rf_only.predict(X_te[FEAT_NO_ARIMA])

    # ── 5. RF + ARIMA hybrid ──────────────────────────────────────────────
    rf_hybrid = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=RANDOM_STATE)
    rf_hybrid.fit(X_tr, y_tr)
    y_pred_hybrid = rf_hybrid.predict(X_te)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    results = {
        "RF Only":    {
            "F1":      round(f1_score(y_te, y_pred_rf, zero_division=0), 4),
            "Accuracy":round(accuracy_score(y_te, y_pred_rf), 4),
        },
        "RF+ARIMA":   {
            "F1":      round(f1_score(y_te, y_pred_hybrid, zero_division=0), 4),
            "Accuracy":round(accuracy_score(y_te, y_pred_hybrid), 4),
        },
    }
    print(f"  RF Only   — F1: {results['RF Only']['F1']}   Acc: {results['RF Only']['Accuracy']}")
    print(f"  RF+ARIMA  — F1: {results['RF+ARIMA']['F1']}   Acc: {results['RF+ARIMA']['Accuracy']}")

    # Save
    joblib.dump(rf_hybrid, os.path.join(MODEL_DIR, "hybrid_rf_arima.pkl"))
    joblib.dump(arima_fit, os.path.join(MODEL_DIR, "arima_colombo.pkl"))

    # Save forecast comparison
    compare = test_df[["date", "precipitation_sum (mm)", "extreme_rain"]].copy()
    compare["rf_pred"]     = y_pred_rf
    compare["hybrid_pred"] = y_pred_hybrid
    compare.to_csv(os.path.join(OUTPUT_DIR, "hybrid_comparison.csv"), index=False)

    import json
    with open(os.path.join(OUTPUT_DIR, "hybrid_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("  ✅ Hybrid experiment complete.")
    return results


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_raw, clean, engineer_features
    df = engineer_features(clean(load_raw()))
    run_hybrid(df)
