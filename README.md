# 🌦️ Predicting Extreme Weather Days in Sri Lanka Using Ensemble Learning

🌐 **Live Dashboard:** https://sri-lanka-weather-ensemble-5upywxdgbpcg2tnfqukvub.streamlit.app/
📁 **GitHub:** https://github.com/Sulakshani/sri-lanka-weather-ensemble

---

## 📌 Project Overview

This project predicts whether a given day in a Sri Lankan district will be an **Extreme Weather Day**
using Ensemble Learning methods applied to 14 years of district-level weather data (2010–2023).

**Definition of Extreme Weather Day:**
- Heavy rainfall: precipitation ≥ 22.3 mm (95th percentile)
- Heatwave: maximum temperature ≥ 34.6°C (95th percentile)

---

## 🎯 Objective

Compare Bagging, Boosting, and Stacking ensemble methods for binary classification of extreme
weather events across all 27 Sri Lankan districts.

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Source | Open-Meteo via Kaggle (tharindumadhusanka9) |
| Records | 142,371 daily observations |
| Districts | 25 (official Sri Lanka districts) |
| Period | January 2010 – September 2023 |
| Target | Extreme Weather Day (Binary: 1/0) |
| Class balance | ~10.2% extreme, ~89.8% normal |

---

## 🤖 Models Implemented

| Model | Method | Type |
|-------|--------|------|
| Random Forest | Bagging | Classification |
| XGBoost | Boosting | Classification |
| LightGBM | Boosting | Classification |
| Stacking (RF+XGB+LGBM → Logistic) | Stacking | Classification |
| RF + ARIMA (Colombo) | Hybridization | Time-series + Classification |

---

## 📊 Actual Test Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Majority Baseline | 0.8978 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Random Forest | 0.9236 | 0.5782 | 0.9344 | 0.7143 | 0.9809 |
| **XGBoost** ⭐ | **0.9453** | **0.6674** | **0.9257** | **0.7756** | **0.9865** |
| LightGBM | 0.9399 | 0.6407 | 0.9386 | 0.7615 | 0.9862 |
| Stacking | 0.9268 | 0.5883 | 0.9441 | 0.7249 | 0.9840 |

**Hybrid (Colombo district):** RF Only F1=0.33 → RF+ARIMA F1=0.80

🏆 **Best Model: XGBoost** (F1=0.7756, ROC-AUC=0.9865)

---

## 🛡️ Data Leakage Prevention

- `precipitation_sum` and `temperature_2m_max` define the target — **excluded from features**
- Rolling/lag features use only past observations (`.shift(1)`)
- Stacking meta-model trained via 5-fold cross-validation (out-of-fold predictions)
- Train/val/test split is stratified — no future data leaks into training

---

## 📁 Project Structure

```
weather_project/
├── app.py                          # Streamlit dashboard
├── data/
│   ├── weatherData.csv
│   └── locationData.csv
├── notebooks/
│   └── ensemble_learning_assignment.ipynb
├── src/
│   ├── preprocess.py               # Data loading, cleaning, feature engineering
│   ├── train.py                    # Model training
│   ├── evaluate.py                 # Model evaluation
│   ├── features.py                 # RF+ARIMA hybrid experiment
│   └── predict.py                  # Prediction interface
├── models/                         # Saved model files (.pkl)
├── outputs/                        # Results CSV, feature importance, hybrid results
├── report/
│   ├── project_summary.md
│   ├── presentation_script.md
│   └── ai_usage_statement.md
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Running

### 1. Create Virtual Environment (PowerShell)
```powershell
cd weather_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models
```powershell
python src/train.py
```

### 3. Evaluate Models
```powershell
python src/evaluate.py
```

### 4. Run Hybrid Experiment
```powershell
python src/features.py
```

### 5. Launch Dashboard
```powershell
streamlit run app.py
```

### 6. Open Notebook
```powershell
jupyter notebook notebooks/ensemble_learning_assignment.ipynb
```

---

## 📖 Bagging vs Boosting vs Stacking

| Method | How it Works | Strength |
|--------|-------------|----------|
| **Bagging** | Parallel trees on random subsets; majority vote | Reduces variance |
| **Boosting** | Sequential trees; each corrects previous errors | Reduces bias |
| **Stacking** | Base models → meta-model combines predictions | Leverages model diversity |

---

## ⚠️ Limitations

- Trained on 2010–2023; future climate shifts may reduce accuracy
- Features use previous-day observations (not real-time)
- ARIMA hybrid tested on Colombo only
- Sub-district spatial variation not captured

---

## 🤖 AI Usage Disclosure

See `report/ai_usage_statement.md`
