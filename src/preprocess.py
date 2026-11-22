"""
preprocess.py
Data loading, cleaning, feature engineering, and splitting
for Sri Lanka Extreme Weather Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ── Thresholds for "extreme" label ──────────────────────────────────────────
RAIN_THRESHOLD = 22.3   # 95th percentile of precipitation_sum (mm)
TEMP_THRESHOLD = 34.6   # 95th percentile of temperature_2m_max (°C)

RANDOM_STATE = 42


def load_raw(data_dir: str = "data") -> pd.DataFrame:
    """Load and merge weatherData + locationData."""
    wx  = pd.read_csv(os.path.join(data_dir, "weatherData.csv"))
    loc = pd.read_csv(os.path.join(data_dir, "locationData.csv"))

    # Parse date
    wx["date"] = pd.to_datetime(wx["date"], format="%m/%d/%Y")

    # Fix district names — map to official 25 Sri Lanka districts
    DISTRICT_MAP = {
        "Kilinochchi[1]": "Kilinochchi",
        "Welimada":       "Badulla",    # town inside Badulla district
        "Bandarawela":    "Badulla",    # town inside Badulla district
    }
    loc["city_name"] = loc["city_name"].replace(DISTRICT_MAP)

    # Merge district name
    df = wx.merge(loc[["location_id", "city_name", "latitude", "longitude"]],
                  on="location_id", how="left")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and validate values."""
    df = df.drop_duplicates(subset=["location_id", "date"]).copy()
    df = df[df["temperature_2m_max (°C)"] >= df["temperature_2m_min (°C)"]].copy()
    df = df[df["precipitation_sum (mm)"] >= 0].copy()
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal + rolling features; create binary extreme label."""
    df = df.sort_values(["location_id", "date"]).copy()

    # ── Temporal ──────────────────────────────────────────────────────────
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["day_of_year"]= df["date"].dt.dayofyear
    df["season"]     = df["month"].map({
        12:"NE Monsoon", 1:"NE Monsoon", 2:"NE Monsoon",
        3:"1st Inter-monsoon", 4:"1st Inter-monsoon", 5:"1st Inter-monsoon",  # noqa
        6:"SW Monsoon", 7:"SW Monsoon", 8:"SW Monsoon",
        9:"2nd Inter-monsoon", 10:"2nd Inter-monsoon", 11:"2nd Inter-monsoon"
    })

    # ── Rolling lags (per location) ───────────────────────────────────────
    for col, alias in [
        ("precipitation_sum (mm)", "rain"),
        ("temperature_2m_max (°C)", "tmax"),
        ("wind_speed_10m_max (km/h)", "wind"),
    ]:
        g = df.groupby("location_id")[col]
        df[f"{alias}_lag1"]    = g.shift(1)
        df[f"{alias}_lag3"]    = g.shift(3)
        df[f"{alias}_roll7"]   = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df[f"{alias}_roll30"]  = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())

    # ── Sunrise/sunset duration ───────────────────────────────────────────
    def parse_hhmm(s):
        try:
            h, m = s.split(":")
            return int(h) * 60 + int(m)
        except Exception:
            return np.nan

    df["sunrise_min"] = df["sunrise"].apply(parse_hhmm)
    df["sunset_min"]  = df["sunset"].apply(parse_hhmm)
    df["daylight_min"]= df["sunset_min"] - df["sunrise_min"]

    # ── Encode district ───────────────────────────────────────────────────
    le = LabelEncoder()
    df["district_enc"] = le.fit_transform(df["city_name"].fillna("Unknown"))

    # ── Target variable ───────────────────────────────────────────────────
    df["extreme"] = (
        (df["precipitation_sum (mm)"] >= RAIN_THRESHOLD) |
        (df["temperature_2m_max (°C)"] >= TEMP_THRESHOLD)
    ).astype(int)

    # Sub-labels (for dashboard explanation)
    df["extreme_rain"] = (df["precipitation_sum (mm)"] >= RAIN_THRESHOLD).astype(int)
    df["extreme_heat"] = (df["temperature_2m_max (°C)"] >= TEMP_THRESHOLD).astype(int)

    return df


def select_features(df: pd.DataFrame):
    """Return feature matrix X and target y, dropping leakage columns."""
    FEATURE_COLS = [
        # Core weather measurements
        # NOTE: precipitation_sum, rain_sum, temperature_2m_max are EXCLUDED
        # because they directly define the target (leakage prevention).
        # Models must learn from lag/rolling features + other predictors.
        "temperature_2m_min (°C)", "temperature_2m_mean (°C)",
        "apparent_temperature_max (°C)", "apparent_temperature_min (°C)",
        "apparent_temperature_mean (°C)",
        "precipitation_hours (h)",
        "wind_speed_10m_max (km/h)", "wind_gusts_10m_max (km/h)",
        "wind_direction_10m_dominant (°)",
        "shortwave_radiation_sum (MJ/m²)", "et0_fao_evapotranspiration (mm)",
        "sunshine_duration (s)", "daylight_duration (s)",
        "weather_code (wmo code)",
        # Engineered
        "month", "day_of_year", "year",
        "rain_lag1", "rain_lag3", "rain_roll7", "rain_roll30",
        "tmax_lag1", "tmax_lag3", "tmax_roll7", "tmax_roll30",
        "wind_lag1", "wind_lag3", "wind_roll7", "wind_roll30",
        "daylight_min", "district_enc",
        "latitude", "longitude",
    ]
    # Keep only columns that exist
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

    df_clean = df.dropna(subset=FEATURE_COLS + ["extreme"]).copy()
    X = df_clean[FEATURE_COLS]
    y = df_clean["extreme"]
    meta = df_clean[["date", "city_name", "location_id", "season",
                      "extreme_rain", "extreme_heat"]].copy()
    return X, y, meta, FEATURE_COLS


def split_data(X, y, meta):
    """Stratified 70/15/15 train/val/test split."""
    X_tv, X_test, y_tv, y_test, m_tv, m_test = train_test_split(
        X, y, meta, test_size=0.15, random_state=RANDOM_STATE, stratify=y)

    X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
        X_tv, y_tv, m_tv, test_size=0.15/0.85, random_state=RANDOM_STATE, stratify=y_tv)

    return X_train, X_val, X_test, y_train, y_val, y_test, m_train, m_val, m_test


def scale_features(X_train, X_val, X_test, out_dir="models"):
    """Fit StandardScaler on train only; return scaled arrays."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    return X_train_s, X_val_s, X_test_s, scaler


def run_preprocessing(data_dir="data", out_dir="models"):
    """Full pipeline — returns all splits + feature names."""
    print("Loading data …")
    df_raw = load_raw(data_dir)
    print(f"  Raw shape: {df_raw.shape}")

    print("Cleaning …")
    df = clean(df_raw)
    print(f"  After clean: {df.shape}")

    print("Engineering features …")
    df = engineer_features(df)

    print("Selecting features …")
    X, y, meta, feat_cols = select_features(df)
    print(f"  X shape: {X.shape}  |  extreme rate: {y.mean()*100:.1f}%")

    print("Splitting …")
    X_train, X_val, X_test, y_train, y_val, y_test, m_train, m_val, m_test = split_data(X, y, meta)
    print(f"  Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

    print("Scaling …")
    X_train_s, X_val_s, X_test_s, scaler = scale_features(
        X_train, X_val, X_test, out_dir)

    # Save processed data for notebook/dashboard
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/processed_data.csv", index=False)

    return dict(
        df=df,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val,   y_test=y_test,
        X_train_s=X_train_s, X_val_s=X_val_s, X_test_s=X_test_s,
        m_train=m_train, m_val=m_val, m_test=m_test,
        feat_cols=feat_cols, scaler=scaler
    )


if __name__ == "__main__":
    run_preprocessing()
