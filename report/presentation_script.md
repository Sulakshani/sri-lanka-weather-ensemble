# 3-Minute Presentation Script
# Predicting Extreme Weather Days in Sri Lanka — Ensemble Learning

---
## BEFORE YOU RECORD — SETUP CHECKLIST
- [ ] Run: streamlit run app.py  →  browser open at localhost:8501
- [ ] Dashboard showing the Overview page
- [ ] Script open on your phone or a second screen
- [ ] Quiet room, mic tested
- [ ] Screen recorder ready (Win+G or OBS)
- [ ] Timer visible so you stay under 3 minutes

---

## FULL SCRIPT WITH WHAT TO SHOW

---

### [0:00 – 0:20] INTRODUCTION
**SHOW: Overview page — just loaded, KPI cards visible**

Say:
"Hi, my name is [your name]. For this Ensemble Learning assignment,
I chose to work on a weather prediction problem using real Sri Lankan data.

My project is — Predicting Extreme Weather Days in Sri Lanka.
The goal is to predict whether a given day in a district will have
extreme weather — either very heavy rainfall or a heatwave."

---

### [0:20 – 0:45] DATASET
**SHOW: Scroll down slowly to the district bar chart**

Say:
"I used a dataset with daily weather readings from all 25 official
Sri Lankan districts, covering 14 years — from 2010 to 2023.
That gives us over 142,000 daily records.

I defined an extreme day as: rainfall above 22.3 millimetres,
or maximum temperature above 34.6 degrees — these are the 95th
percentile values from the actual data.

About 10 percent of days are extreme, which makes this an
imbalanced classification problem."

**SHOW: Point at the bar chart — Ratnapura, Kalutara high bars**

Say:
"You can see that Wet Zone districts like Ratnapura and Kalutara
have much higher extreme weather rates — which matches what we
know about Sri Lanka's climate."

---

### [0:45 – 1:15] WHAT I BUILT — MODELS
**SHOW: Click to Model Comparison page — metric cards visible**

Say:
"I implemented five models across three ensemble paradigms.

For Bagging — I used Random Forest with 200 decision trees.
Each tree trains on a random subset of data and they vote together.
I tuned it using GridSearchCV to find the best depth and tree count.

For Boosting — I compared XGBoost, LightGBM, and CatBoost.
These build trees one at a time, each one correcting the mistakes
of the previous. This is very powerful for structured tabular data.

For Stacking — I combined Random Forest, XGBoost, and LightGBM
as base models, with Logistic Regression as the meta-learner.
I used 5-fold cross-validation to make sure no data leaks into
the meta-model.

Because only 10 percent of days are extreme, I used SMOTE —
Synthetic Minority Over-sampling — to balance the training data
before fitting any model."

---

### [1:15 – 1:45] DASHBOARD DEMO
**SHOW: Stay on Model Comparison — point at the metric cards**

Say:
"This is the main comparison page. Each card shows the test set
performance for one model — F1 score, ROC-AUC, precision and recall."

**SHOW: Click the Bar Charts tab — select F1 Score**

Say:
"CatBoost achieved the highest F1 of 0.8056. XGBoost and LightGBM
are very close behind. Random Forest is slightly lower, and Stacking
sits between them."

**SHOW: Click the Radar Chart tab**

Say:
"The radar chart shows all five metrics at once. The boosting models
have the largest area — consistently strong across every metric."

**SHOW: Click Feature Importance tab — select XGBoost**

Say:
"Feature importance tells us which variables matter most.
The 7-day rolling rainfall, precipitation hours, and apparent
temperature are the strongest predictors — which makes sense.
Recent rain history is a very strong signal for the next extreme day."

---

### [1:45 – 2:10] HYBRIDIZATION
**SHOW: Click the RF+ARIMA Hybrid tab**

Say:
"My lecturer specifically asked for a hybridization experiment.
For time-series weather data, the recommended approach was
Random Forest combined with ARIMA.

Here is how I did it: I fitted an ARIMA model on Colombo's
historical daily rainfall to capture the long-term trend.
Then I calculated the residuals — the parts ARIMA could not explain.
I gave those residuals as an extra feature to Random Forest,
so it could learn from unexpected rainfall spikes.

The result — Random Forest alone scored F1 of 0.33 on the
Colombo test period. With ARIMA residuals added, it jumped to 0.80.
That is a very significant improvement."

---

### [2:10 – 2:30] PREDICTION PAGE
**SHOW: Click to Prediction page**

Say:
"The dashboard also has a live prediction page.
You enter yesterday's weather conditions — temperature, wind,
precipitation hours — choose a model, and click Predict.

The gauge shows the probability of an extreme weather event,
and the table below compares predictions from all five models
at the same time."

**SHOW: Enter a few values — set rain_lag1 to 30, prec hours to 8 — click Predict**

Say:
"With high rainfall history and many precipitation hours,
most models predict an extreme day — which is exactly
what we would expect."

---

### [2:30 – 2:50] RESULTS & WHY
**SHOW: Go back to Model Comparison — metric cards visible**

Say:
"So why does CatBoost perform best?

CatBoost uses symmetric trees and built-in regularisation,
which helps it handle the boundary between normal and extreme days
very precisely. After SMOTE balancing, all three boosting models
perform similarly — the differences are small.

Stacking does not outperform the best individual model here,
because when base models are already strong and well-calibrated,
the meta-learner gains very little from combining them.

All models massively outperform the baseline — which always
predicts Normal and scores F1 of zero."

---

### [2:50 – 3:00] CLOSE
**SHOW: Overview page**

Say:
"To summarise — I built a complete Ensemble Learning pipeline
on real Sri Lankan weather data, comparing Bagging, Boosting,
and Stacking, with SMOTE for imbalance and GridSearchCV for tuning.

The RF plus ARIMA hybrid showed a strong improvement on the
time-series experiment. The full code and dashboard are on GitHub.

Thank you."

---

## CLICK ORDER SUMMARY
1. Overview — KPI cards and bar chart
2. Model Comparison → Bar Charts tab (F1 Score)
3. Model Comparison → Radar Chart tab
4. Model Comparison → Feature Importance tab
5. Model Comparison → RF+ARIMA Hybrid tab
6. Prediction — enter values, click Predict, show gauge
7. Back to Model Comparison — metric cards
8. Back to Overview — close

---

## TIPS TO SOUND NATURAL
- Read the script 2-3 times before recording so you know it
- Speak slowly — nerves make people rush
- It is fine to pause while you click — silence is better than filler words
- If you make a mistake, pause 3 seconds and repeat the sentence — easy to trim in editing
- Do not read word-for-word in a flat voice — look at the screen and describe what you see

---

## WHAT TO SAY IF ASKED "DID YOU USE AI?"
"Yes — I used Claude to help generate the dashboard code and
structure the pipeline. All the model training, evaluation,
and results are from my own runs on the actual dataset.
I understand every part of the code and can explain the
decisions I made — like why I chose SMOTE, why I excluded
same-day rainfall from features to prevent leakage,
and how I set the 95th percentile thresholds."

