"""
train.py
Train all ensemble models with:
  - SMOTE for class imbalance handling
  - GridSearchCV for hyperparameter tuning (RF + XGBoost)
  - Random Forest     (Bagging)
  - XGBoost           (Boosting)
  - LightGBM          (Boosting)
  - CatBoost          (Boosting)
  - Stacking Ensemble (RF + XGB + LGBM → Logistic Regression)
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from preprocess import run_preprocessing

RANDOM_STATE = 42
MODEL_DIR    = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ── SMOTE ─────────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    """
    Apply SMOTE to oversample the minority class (extreme=1).
    SMOTE generates synthetic samples so the classifier sees a balanced dataset.
    Fit only on training data — never on val/test.
    """
    print(f"  Before SMOTE — Class 0: {(y_train==0).sum()}  Class 1: {(y_train==1).sum()}")
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After  SMOTE — Class 0: {(y_res==0).sum()}  Class 1: {(y_res==1).sum()}")
    return X_res, y_res


# ── 1. Random Forest with GridSearchCV ───────────────────────────────────────
def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n[1/5] Random Forest (Bagging) + GridSearchCV …")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [10, 15],
        "min_samples_leaf": [3, 5],
    }
    base_rf = RandomForestClassifier(
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    gs = GridSearchCV(
        base_rf, param_grid,
        scoring=make_scorer(f1_score),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Val F1: {f1_score(y_val, best.predict(X_val)):.4f}")
    return best


# ── 2. XGBoost with GridSearchCV ─────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n[2/5] XGBoost (Boosting) + GridSearchCV …")

    # Note: SMOTE already balances classes, so scale_pos_weight=1
    param_grid = {
        "n_estimators":    [200, 300],
        "max_depth":       [5, 6],
        "learning_rate":   [0.05, 0.08],
    }
    base_xgb = xgb.XGBClassifier(
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0
    )
    gs = GridSearchCV(
        base_xgb, param_grid,
        scoring=make_scorer(f1_score),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")
    print(f"  Val F1: {f1_score(y_val, best.predict(X_val)):.4f}")
    return best


# ── 3. LightGBM ───────────────────────────────────────────────────────────────
def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\n[3/5] LightGBM (Boosting) …")
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    # SMOTE balances classes — no scale_pos_weight needed
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    print(f"  Val F1: {f1_score(y_val, model.predict(X_val)):.4f}")
    return model


# ── 4. CatBoost ───────────────────────────────────────────────────────────────
def train_catboost(X_train, y_train, X_val, y_val):
    print("\n[4/5] CatBoost (Boosting) …")
    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.08,
        eval_metric="F1", random_seed=RANDOM_STATE,
        verbose=0, early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    print(f"  Val F1: {f1_score(y_val, model.predict(X_val)):.4f}")
    return model


# ── 5. Stacking ───────────────────────────────────────────────────────────────
def train_stacking(X_train, y_train, X_val, y_val):
    print("\n[5/5] Stacking Ensemble (RF + XGB + LGBM → Logistic Regression) …")
    estimators = [
        ("rf",   RandomForestClassifier(
                    n_estimators=100, max_depth=10, max_features="sqrt",
                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ("xgb",  xgb.XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, n_jobs=-1,
                    verbosity=0, eval_metric="logloss")),
        ("lgbm", lgb.LGBMClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]
    meta = LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_STATE)
    stacker = StackingClassifier(
        estimators=estimators, final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        passthrough=False, n_jobs=1
    )
    stacker.fit(X_train, y_train)
    print(f"  Val F1: {f1_score(y_val, stacker.predict(X_val)):.4f}")
    return stacker


# ── Main ──────────────────────────────────────────────────────────────────────
def train_all():
    data = run_preprocessing()

    X_train_raw = data["X_train"].values
    X_val_raw   = data["X_val"].values
    X_test_raw  = data["X_test"].values
    y_train     = data["y_train"]
    y_val       = data["y_val"]

    # ── Apply SMOTE on training set only ──────────────────────────────────
    print("\n── Applying SMOTE to training data ──")
    X_train_sm, y_train_sm = apply_smote(X_train_raw, y_train)

    # ── Train all models on SMOTE-balanced data ───────────────────────────
    rf      = train_random_forest(X_train_sm, y_train_sm, X_val_raw, y_val)
    xgbm    = train_xgboost(X_train_sm, y_train_sm, X_val_raw, y_val)
    lgbm    = train_lightgbm(X_train_sm, y_train_sm, X_val_raw, y_val)
    catb    = train_catboost(X_train_sm, y_train_sm, X_val_raw, y_val)
    stack   = train_stacking(X_train_sm, y_train_sm, X_val_raw, y_val)

    # ── Save all models ───────────────────────────────────────────────────
    joblib.dump(rf,    os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(xgbm,  os.path.join(MODEL_DIR, "xgboost.pkl"))
    joblib.dump(lgbm,  os.path.join(MODEL_DIR, "lightgbm.pkl"))
    joblib.dump(catb,  os.path.join(MODEL_DIR, "catboost.pkl"))
    joblib.dump(stack, os.path.join(MODEL_DIR, "stacking.pkl"))
    joblib.dump(data["feat_cols"], os.path.join(MODEL_DIR, "feature_cols.pkl"))

    print("\n✅ All 5 models saved to models/")
    return data


if __name__ == "__main__":
    train_all()
