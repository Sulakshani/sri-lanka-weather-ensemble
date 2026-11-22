# Project Summary
## Predicting Extreme Weather Days in Sri Lanka Using Ensemble Learning

**Student:** P. D. S. C. Dissanayake


---

## Problem Statement

Sri Lanka is frequently affected by extreme weather events — heavy monsoon rainfall causing floods,
and heat waves affecting agriculture and public health. This project builds a machine learning
system to predict whether a given day in a Sri Lankan district will be an **Extreme Weather Day**.

An Extreme Weather Day is defined as:
- **Heavy Rainfall:** precipitation ≥ 22.3 mm (95th percentile of all daily values)
- **Heatwave:** maximum temperature ≥ 34.6°C (95th percentile)

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | Open-Meteo historical data via Kaggle |
| Coverage | 25 districts, 2010–2023 |
| Total records | 142,371 daily rows |
| Target class balance | 10.2% extreme, 89.8% normal |
| Missing values | None |

**Key features used (after leakage removal):**
- Previous-day temperature, apparent temperature, wind speed
- Precipitation hours, sunshine duration, solar radiation
- 1-day, 3-day, 7-day, 30-day rolling rainfall and temperature
- Month, day of year, district encoding, coordinates

**Excluded from features (leakage prevention):**
- `precipitation_sum` — directly defines the rainfall extreme label
- `temperature_2m_max` — directly defines the heat extreme label

---

## Methodology

### Data Split
- Train: 99,602 rows (70%)
- Validation: 21,344 rows (15%)
- Test: 21,344 rows (15%)
- Stratified split to preserve 10.2% extreme class rate in all sets

### Models

**1. Random Forest (Bagging)**
- 200 decision trees, max depth 12
- Bootstrap sampling with max_features="sqrt"
- class_weight="balanced" to handle imbalance

**2. XGBoost (Boosting)**
- 300 trees, learning rate 0.08, max depth 6
- scale_pos_weight to handle imbalance (~8.8×)
- Trained with validation set for monitoring

**3. LightGBM (Boosting)**
- 300 trees, early stopping on validation
- Same hyperparameter structure as XGBoost

**4. Stacking Ensemble**
- Base learners: RF + XGBoost + LightGBM
- Meta-model: Logistic Regression
- 5-fold StratifiedKFold cross-validation for leakage-safe meta-features

**5. RF + ARIMA Hybrid (Colombo)**
- ARIMA(2,1,2) fitted on Colombo training rainfall series
- RF learns from ARIMA residuals + lag features
- Compared against RF using only lag features (no ARIMA)

---

## Results

| Model | F1 Score | ROC-AUC | Method |
|-------|----------|---------|--------|
| Random Forest | 0.7668 | 0.9813 | Bagging + GridSearchCV |
| XGBoost | 0.8061 | 0.9862 | Boosting + GridSearchCV |
| **LightGBM** ⭐ | **0.8068** | **0.9858** | Boosting + early stopping |
| CatBoost | 0.8039 | 0.9848 | Boosting |
| Stacking | 0.7847 | 0.9814 | RF+XGB+LGBM → Logistic |
| RF+ARIMA hybrid | ~0.798 | — | Time-series hybridization |

---

## Key Conclusions

1. **XGBoost is the best model** (F1=0.7756). Sequential boosting helps it learn the
   precise boundary between normal and extreme days better than bagging or stacking.

2. **All models vastly outperform the baseline** (F1=0 → 0.72–0.78), confirming the
   models genuinely learn weather patterns from the data.

3. **Stacking does not outperform XGBoost** because the base models are already strong;
   the meta-model gains little from combining well-calibrated predictions.

4. **RF+ARIMA hybrid improves F1 from 0.33 to 0.80** on the Colombo rainfall time series,
   confirming that ARIMA residuals carry useful signal for extreme event detection.

5. **High ROC-AUC (~0.98–0.99)** across all models shows excellent discrimination ability,
   even though Recall is prioritized over Precision in this application (missing an extreme
   event is more costly than a false alarm).

---

## Research Comparison

| Model | Type | Strength | Best For |
|-------|------|----------|----------|
| Random Forest | Bagging | Stable, interpretable | Noisy tabular data |
| XGBoost | Boosting | High accuracy, handles imbalance | Competitions, tabular |
| LightGBM | Boosting | Faster than XGBoost | Large datasets |
| CatBoost | Boosting | Handles categorical natively | Mixed-type data |

---

## Limitations & Future Work

- Models trained on 2010–2023; climate shift post-training may reduce accuracy
- Features use previous-day observations — not real-time weather radar
- ARIMA hybrid only tested on Colombo; could be extended to all districts
- Future work: LSTM or Transformer models for sequence modeling; ensemble of ARIMA per district
