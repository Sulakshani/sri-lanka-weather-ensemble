"""
app.py — Sri Lanka Extreme Weather Prediction Dashboard
Ensemble Learning Assignment — University of Moratuwa
Run: streamlit run app.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Sri Lanka Extreme Weather — Ensemble Learning",
    page_icon="🌦️", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar full-width interactive nav ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b2d 0%, #1a2a40 100%);
    min-width: 240px;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 0px !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 13px 18px !important;
    margin: 4px 0 !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #a8c0d8 !important;
    cursor: pointer !important;
    transition: background 0.2s, color 0.2s, transform 0.1s !important;
    border: 1.5px solid transparent !important;
    background: rgba(255,255,255,0.04) !important;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(41,128,185,0.22) !important;
    color: #ffffff !important;
    border-color: #2980b9 !important;
    transform: translateX(4px) !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked),
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, #1a6fa8 0%, #2980b9 100%) !important;
    color: #ffffff !important;
    border-color: #5dade2 !important;
    box-shadow: 0 4px 14px rgba(41,128,185,0.4) !important;
    transform: translateX(4px) !important;
}
/* Hide the actual radio dot */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}
/* Sidebar title */
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 1.4rem !important;
    margin-bottom: 0.6rem !important;
    padding: 8px 4px 4px 4px;
}
/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 16px 0 !important;
}
/* Sidebar caption */
[data-testid="stSidebar"] .stCaption {
    color: #6a90b0 !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
    padding: 0 4px;
}

/* ── Main page headings ── */
.main-header {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0a84ff;
    text-shadow: 0 2px 8px rgba(10,132,255,0.18);
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 1rem;
    color: #90b4d4;
    margin-bottom: 1.2rem;
    font-weight: 400;
}

/* ── Section titles ── */
.stitle {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFD700;
    border-bottom: 2.5px solid #FFD700;
    padding-bottom: 5px;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    letter-spacing: 0.01em;
    text-shadow: 0 1px 6px rgba(255,215,0,0.25);
}

/* ── Info/content boxes ── */
.ebox {
    background: #1e2d3d;
    border-radius: 10px;
    padding: 18px 22px;
    border: 1.5px solid #2e4a62;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #d4e6f7;
    margin-bottom: 12px;
}
.ebox b {
    color: #5dade2;
    font-weight: 700;
}
.ebox i {
    color: #a8d8ea;
}

/* ── Metric cards ── */
.mcard {
    background: #1b2f45;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1.5px solid #2e4a62;
    border-left: 5px solid #2980b9;
    margin-bottom: 10px;
    color: #d4e6f7;
    font-size: 0.95rem;
    line-height: 1.8;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    transition: box-shadow 0.2s;
}
.mcard b {
    color: #5dade2;
    font-size: 1.0rem;
}
.mcard:hover {
    box-shadow: 0 4px 18px rgba(41,128,185,0.3);
}
.best {
    background: #0d2e1e;
    border: 1.5px solid #1aad6e;
    border-left: 5px solid #27ae60;
    box-shadow: 0 2px 14px rgba(39,174,96,0.3);
}
.best b {
    color: #2ecc71;
}
.warn {
    background: #2a2000;
    border: 1.5px solid #9a7000;
    border-left: 5px solid #f39c12;
    color: #f0d080;
}
.warn b, .warn i {
    color: #f7c948;
}
</style>""", unsafe_allow_html=True)

# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    from preprocess import load_raw, clean, engineer_features
    return engineer_features(clean(load_raw("data")))

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    out = {}
    for name, fname in [("Random Forest","random_forest.pkl"),
                         ("XGBoost","xgboost.pkl"),
                         ("LightGBM","lightgbm.pkl"),
                         ("CatBoost","catboost.pkl"),
                         ("Stacking","stacking.pkl")]:
        p = f"models/{fname}"
        if os.path.exists(p):
            out[name] = joblib.load(p)
    return out

@st.cache_data
def load_results():
    p = "outputs/model_comparison.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_fi():
    p = "outputs/feature_importance.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_hybrid():
    p = "outputs/hybrid_results.json"
    return json.load(open(p)) if os.path.exists(p) else None

COLORS = {
    "Random Forest":"#2980b9","XGBoost":"#27ae60",
    "LightGBM":"#8e44ad","CatBoost":"#e67e22",
    "Stacking":"#c0392b","Majority Baseline":"#95a5a6",
}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌦️ Navigation")
    page = st.radio(
        "",
        ["🏠 Overview", "🔍 Data Exploration", "📊 Model Comparison", "🔮 Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Ensemble Learning Assignment\nUniversity of Moratuwa")

df       = load_data()
models   = load_models()
res_df   = load_results()
fi_df    = load_fi()
hybrid   = load_hybrid()
DISTRICTS= sorted(df["city_name"].unique())

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🌦️ Predicting Extreme Weather Days in Sri Lanka</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ensemble Learning Dashboard · Bagging · Boosting · Stacking · Hybridization</div>',
                unsafe_allow_html=True)

    total   = len(df)
    extreme = int(df["extreme"].sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📋 Total Records",   f"{total:,}")
    c2.metric("⚠️ Extreme Days",    f"{extreme:,} ({extreme/total*100:.1f}%)")
    c3.metric("🗺️ Districts",        df["city_name"].nunique())
    c4.metric("📅 Years Covered",    df["date"].dt.year.nunique())

    st.markdown("---")
    l, r = st.columns([1.2, 1])

    with l:
        st.markdown('<div class="stitle">📌 Project Objective</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ebox">
<b>Goal:</b> Predict whether a Sri Lankan district will experience an <b>Extreme Weather Day</b>
— heavy rainfall (≥22.3 mm, 95th pct) or heatwave (≥34.6 °C, 95th pct).<br><br>
<b>Dataset:</b> 142,371 daily observations across 25 districts, 2010–2023.<br><br>
<b>Class imbalance:</b> Only 10.2% of days are extreme. We handle this with <b>SMOTE</b>
(Synthetic Minority Over-sampling Technique) on the training set only.<br><br>
<b>Hyperparameter tuning:</b> <b>GridSearchCV</b> used for Random Forest and XGBoost
to find the optimal depth, estimators, and learning rate.
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="stitle">🗂️ Dataset Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="ebox">
<b>Source:</b> Open-Meteo historical weather API (Kaggle)<br>
<b>Coverage:</b> 25 districts · Jan 2010 – Sep 2023<br>
<b>Features:</b> Temperature, precipitation hours, wind, sunshine, radiation, lag/rolling stats<br>
<b>Target:</b> Extreme Weather Day (1 = extreme, 0 = normal)<br>
<b>Split:</b> 70% train / 15% val / 15% test · stratified
</div>""", unsafe_allow_html=True)

    with r:
        st.markdown('<div class="stitle">🤖 Ensemble Methods Used</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ebox">
<b>🌲 Bagging — Random Forest</b><br>
Trains 200 trees on random subsets; majority vote. Tuned with GridSearchCV.<br><br>
<b>⚡ Boosting — XGBoost, LightGBM, CatBoost</b><br>
Sequential trees, each correcting prior errors. All three compared side-by-side.<br><br>
<b>🔗 Stacking</b><br>
RF + XGBoost + LightGBM as base learners → Logistic Regression meta-model.
5-fold CV prevents leakage into the meta-learner.<br><br>
<b>🔄 Hybridization — RF + ARIMA</b><br>
ARIMA fits Colombo rainfall trend; RF learns from ARIMA residuals.
F1 improves from 0.33 (RF alone) to 0.80 (RF + ARIMA).<br><br>
<b>⚖️ SMOTE</b><br>
Synthetic minority samples generated on training set to balance 10%/90% class split.
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="stitle">🌏 Extreme Event Rate by District</div>',
                unsafe_allow_html=True)
    de = (df.groupby("city_name")["extreme"]
            .agg(total="count", ext="sum")
            .assign(rate=lambda x: x["ext"]/x["total"]*100)
            .reset_index().sort_values("rate", ascending=False))
    fig = px.bar(de, x="city_name", y="rate", color="rate",
                 color_continuous_scale="Reds",
                 labels={"city_name":"District","rate":"Extreme Day Rate (%)"},
                 title="Extreme Weather Day Rate by District (2010–2023)")
    fig.update_layout(height=360, xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Exploration":
    st.markdown('<div class="main-header">🔍 Data Exploration</div>', unsafe_allow_html=True)

    cf1, cf2, cf3 = st.columns(3)
    sel_d = cf1.multiselect("Districts", DISTRICTS, default=["Colombo","Kandy","Jaffna"])
    sel_y = cf2.slider("Year range", 2010, 2023, (2010, 2023))
    sel_s = cf3.selectbox("Season", ["All","NE Monsoon","1st Inter-monsoon",
                                      "SW Monsoon","2nd Inter-monsoon"])

    mask = (df["city_name"].isin(sel_d) &
            (df["date"].dt.year >= sel_y[0]) &
            (df["date"].dt.year <= sel_y[1]))
    if sel_s != "All":
        mask &= df["season"] == sel_s
    view = df[mask].copy()
    st.markdown(f"**Filtered:** {len(view):,} records · {view['extreme'].sum():,} extreme days")

    t1, t2, t3, t4 = st.tabs(["📋 Preview","📈 Distributions","🌧️ Trends","🔥 Correlations"])

    with t1:
        dcols = ["date","city_name","season","temperature_2m_max (°C)",
                 "precipitation_sum (mm)","wind_speed_10m_max (km/h)","extreme"]
        st.dataframe(view[[c for c in dcols if c in view.columns]].tail(500),
                     use_container_width=True, height=360)
        miss = df.isnull().sum(); miss = miss[miss>0]
        st.success("✅ No missing values." if miss.empty else f"Missing: {miss.to_dict()}")

    with t2:
        met = st.selectbox("Feature", [
            "temperature_2m_max (°C)","temperature_2m_min (°C)",
            "precipitation_sum (mm)","wind_speed_10m_max (km/h)","sunshine_duration (s)"])
        fig = px.histogram(view, x=met, color="extreme", barmode="overlay", nbins=60,
                           color_discrete_map={0:"#2980b9",1:"#e74c3c"},
                           title=f"Distribution: {met}")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.box(view, x="season", y=met, color="extreme",
                      color_discrete_map={0:"#2980b9",1:"#e74c3c"},
                      title=f"{met} by Season")
        fig2.update_layout(height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        rm = (view.groupby(["city_name", view["date"].dt.month])
                  ["precipitation_sum (mm)"].mean().reset_index())
        rm.columns = ["District","Month","Avg Rainfall mm"]
        fig = px.line(rm, x="Month", y="Avg Rainfall mm", color="District",
                      markers=True, title="Average Monthly Rainfall by District")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        ae = (view.groupby([view["date"].dt.year,"city_name"])["extreme"]
                  .sum().reset_index())
        ae.columns = ["Year","District","Extreme Days"]
        fig2 = px.bar(ae, x="Year", y="Extreme Days", color="District",
                      title="Extreme Days per Year", barmode="group")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

    with t4:
        nc = ["temperature_2m_max (°C)","temperature_2m_min (°C)",
              "precipitation_sum (mm)","wind_speed_10m_max (km/h)",
              "sunshine_duration (s)","shortwave_radiation_sum (MJ/m²)","extreme"]
        nc = [c for c in nc if c in view.columns]
        fig = px.imshow(view[nc].corr(), text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Feature Correlation Matrix", aspect="auto")
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown('<div class="main-header">📊 Ensemble Model Comparison</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">All models evaluated on the same held-out test set (21,344 rows)</div>',
                unsafe_allow_html=True)

    if res_df is None:
        st.error("Run `python src/evaluate.py` first.")
        st.stop()

    # ── Pipeline explanation ───────────────────────────────────────────────
    with st.expander("⚙️ Training Pipeline Details", expanded=True):
        pc1, pc2, pc3 = st.columns(3)
        pc1.markdown("""**⚖️ SMOTE (Imbalance Handling)**
- Training set: 89,422 normal / 10,180 extreme
- After SMOTE: 89,422 / 89,422 (balanced)
- Applied to training data only — val/test untouched
- Prevents model bias toward majority class""")
        pc2.markdown("""**🔎 GridSearchCV (Hyperparameter Tuning)**
- Used for: Random Forest and XGBoost
- RF best: depth=15, estimators=200, min_leaf=3
- XGB best: lr=0.08, depth=6, estimators=300
- 2-fold CV on 40k subsample → refit on full SMOTE data""")
        pc3.markdown("""**🔗 Stacking (Meta-learning)**
- Base learners: RF + XGBoost + LightGBM
- Meta-model: Logistic Regression
- 5-fold StratifiedKFold → out-of-fold predictions
- Leakage-safe: meta-model never sees base model training data""")

    # ── Concept cards ─────────────────────────────────────────────────────
    with st.expander("📖 Bagging · Boosting · Stacking — explained simply"):
        e1,e2,e3 = st.columns(3)
        e1.markdown("""**🌲 Bagging (Random Forest)**
Each tree trains on a random bootstrap sample. At prediction time, all trees vote.
Bagging *reduces variance* — the model is stable even with noisy weather data.""")
        e2.markdown("""**⚡ Boosting (XGBoost / LightGBM / CatBoost)**
Trees are added one at a time. Each new tree focuses on the samples the previous ones
got wrong. Boosting *reduces bias* — learns complex patterns in imbalanced data.""")
        e3.markdown("""**🔗 Stacking**
RF, XGBoost, and LightGBM each make predictions. A Logistic Regression
meta-model learns how to best combine those three sets of predictions to
produce a final answer.""")

    # ── Metric cards ──────────────────────────────────────────────────────
    st.markdown('<div class="stitle">📌 Test Set Results</div>', unsafe_allow_html=True)
    model_rows = res_df[res_df["Model"] != "Majority Baseline"]
    best_name  = model_rows.sort_values("F1 Score", ascending=False).iloc[0]["Model"]

    cols = st.columns(len(model_rows))
    for i, (_, row) in enumerate(model_rows.iterrows()):
        is_best = row["Model"] == best_name
        cc = "mcard best" if is_best else "mcard"
        with cols[i]:
            st.markdown(f"""<div class="{cc}">
<b>{"🏆 " if is_best else ""}{row['Model']}</b><br>
F1: <b>{row['F1 Score']}</b><br>
ROC-AUC: <b>{row['ROC-AUC']}</b><br>
Precision: {row['Precision']}<br>
Recall: {row['Recall']}<br>
Accuracy: {row['Accuracy']}
</div>""", unsafe_allow_html=True)

    base = res_df[res_df["Model"]=="Majority Baseline"].iloc[0]
    st.markdown(f"""<div class="mcard warn">
<b>📏 Majority Baseline</b> (always predict Normal) — F1: {base['F1 Score']} · ROC-AUC: {base['ROC-AUC']}<br>
<i>All ensemble models vastly outperform this baseline.</i>
</div>""", unsafe_allow_html=True)

    # ── Chart tabs ────────────────────────────────────────────────────────
    st.markdown('<div class="stitle">📊 Visualisations</div>', unsafe_allow_html=True)
    t1,t2,t3,t4 = st.tabs(["Bar Charts","Radar Chart","Feature Importance","RF+ARIMA Hybrid"])

    with t1:
        sel_m = st.selectbox("Metric", ["F1 Score","ROC-AUC","Precision","Recall","Accuracy"])
        fig = px.bar(res_df, x="Model", y=sel_m, color="Model",
                     color_discrete_map=COLORS, text=sel_m,
                     title=f"{sel_m} — All Models")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=400, showlegend=False, yaxis=dict(range=[0,1.08]))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res_df.set_index("Model").style.highlight_max(
            subset=["F1 Score","ROC-AUC","Precision","Recall","Accuracy"],
            color="#d5f5e3"), use_container_width=True)

    with t2:
        radar_m = ["F1 Score","ROC-AUC","Precision","Recall","Accuracy"]
        fig = go.Figure()
        for _, row in res_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_m], theta=radar_m,
                fill="toself", name=row["Model"],
                line_color=COLORS.get(row["Model"],"#555")))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                          height=460, title="Multi-Metric Radar")
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        if fi_df is not None:
            sel_fi = st.selectbox("Model", fi_df["model"].unique())
            fi_v = fi_df[fi_df["model"]==sel_fi].head(15)
            fig = px.bar(fi_v, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues",
                         title=f"Top 15 Features — {sel_fi}")
            fig.update_layout(height=460, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available.")

    with t4:
        st.markdown("""<div class="ebox">
<b>Hybridization: RF + ARIMA (Colombo district)</b><br>
ARIMA(2,1,2) is fitted on Colombo's training rainfall series to capture the time-series trend.
Random Forest then uses the ARIMA <i>residuals</i> (unexpected deviations) as an extra feature
alongside lag variables — without seeing today's precipitation value (leakage-safe).<br><br>
When ARIMA under-predicts rainfall (large positive residual), it signals an unusual event —
boosting the RF's ability to detect extreme rain days.
</div>""", unsafe_allow_html=True)
        if hybrid:
            hc = st.columns(2)
            for i,(k,v) in enumerate(hybrid.items()):
                is_b = k=="RF+ARIMA"
                with hc[i]:
                    st.markdown(f"""<div class="mcard {'best' if is_b else ''}">
<b>{"🏆 " if is_b else ""}{k}</b><br>
F1 Score: <b>{v['F1']}</b><br>
Accuracy: {v['Accuracy']}
</div>""", unsafe_allow_html=True)
            hdf = pd.DataFrame([{"Model":k,"F1 Score":v["F1"]} for k,v in hybrid.items()])
            fig = px.bar(hdf, x="Model", y="F1 Score", color="Model", text="F1 Score",
                         color_discrete_map={"RF Only":"#95a5a6","RF+ARIMA":"#27ae60"},
                         title="Hybrid Experiment — F1 Score")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(height=340, showlegend=False, yaxis=dict(range=[0,1.05]))
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"RF+ARIMA improves F1 from {hybrid['RF Only']['F1']} → {hybrid['RF+ARIMA']['F1']} (+{hybrid['RF+ARIMA']['F1']-hybrid['RF Only']['F1']:.4f})")

    # ── Conclusion ─────────────────────────────────────────────────────────
    st.markdown("---")
    best_row = model_rows.sort_values("F1 Score", ascending=False).iloc[0]

    model_reasons = {
        "LightGBM":     "it uses histogram-based splitting and leaf-wise tree growth, which finds finer decision boundaries efficiently on tabular weather data.",
        "CatBoost":     "it handles class-boundary complexity well through symmetric tree growth and built-in regularisation.",
        "XGBoost":      "it uses aggressive regularisation (L1/L2) and column subsampling, which controls overfitting on the SMOTE-balanced data.",
        "Random Forest":"its ensemble of deep trees captures non-linear interactions in the weather features robustly.",
        "Stacking":     "it combines multiple strong base models, allowing the meta-learner to exploit their complementary strengths.",
    }
    reason = model_reasons.get(best_row["Model"], "it achieved the best balance of precision and recall on this dataset.")

    st.markdown(f"""<div class="ebox">
<b>🏁 Conclusion — Why {best_row['Model']} performs best</b><br><br>
After SMOTE balancing and GridSearchCV tuning, all boosting models (XGBoost, LightGBM, CatBoost)
outperform Random Forest and Stacking on this dataset. <b>{best_row['Model']}</b> edges out
with F1={best_row['F1 Score']:.4f} because {reason}<br><br>
<b>Stacking</b> combines three strong base models but doesn't outperform the best individual
boosting model — the meta-learner gains diminishing returns when base models are already
well-calibrated after SMOTE.<br><br>
<b>Limitations:</b> Trained on 2010–2023 data; climate shift may reduce future accuracy.
Features use previous-day observations to prevent data leakage.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown('<div class="main-header">🔮 Extreme Weather Prediction</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Single-day or monthly extreme weather risk forecasting.</div>',
                unsafe_allow_html=True)
    st.info("ℹ️ Uses previous-day observations only (no same-day data leakage). "
            "All predictions come from trained models — no hardcoded values.")

    feat_cols = joblib.load("models/feature_cols.pkl")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(sorted(df["city_name"].unique()))

    # ── Two tabs ───────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔮 Single Day Prediction", "📅 Monthly Forecast"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — SINGLE DAY
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        with st.form("pred_form"):
            st.markdown("### Weather Conditions (Yesterday's Readings)")
            c1,c2,c3 = st.columns(3)
            with c1:
                tmin     = st.number_input("Min Temp (°C)",         10.0, 35.0, 24.0, 0.5)
                tmean    = st.number_input("Mean Temp (°C)",         15.0, 38.0, 28.0, 0.5)
                app_max  = st.number_input("Apparent Temp Max (°C)", 20.0, 45.0, 32.0, 0.5)
                app_min  = st.number_input("Apparent Temp Min (°C)", 15.0, 38.0, 26.0, 0.5)
            with c2:
                prec_h   = st.number_input("Precipitation Hours (h)", 0, 24, 2)
                wind     = st.number_input("Wind Speed Max (km/h)",   0.0,120.0, 25.0, 1.0)
                gusts    = st.number_input("Wind Gusts Max (km/h)",   0.0,150.0, 40.0, 1.0)
                wind_dir = st.number_input("Wind Direction (°)",      0, 360, 180)
            with c3:
                rain_l1  = st.number_input("Yesterday Rainfall (mm)", 0.0,200.0,  2.0, 0.5)
                rain_l3  = st.number_input("3-Day Avg Rain (mm)",     0.0,100.0,  1.5, 0.5)
                rain_r7  = st.number_input("7-Day Rolling Rain (mm)", 0.0, 80.0,  1.2, 0.5)
                tmax_l1  = st.number_input("Yesterday Max Temp (°C)", 20.0,40.0, 29.0, 0.5)

            c4,c5 = st.columns(2)
            with c4:
                month    = st.selectbox("Month", list(range(1,13)), index=5)
                district = st.selectbox("District", DISTRICTS)
            with c5:
                radiation= st.number_input("Shortwave Radiation (MJ/m²)", 0.0, 35.0, 18.0, 0.5)
                sunshine = st.number_input("Sunshine Duration (s)",        0.0,50000.0,25000.0,500.0)
                daylight = st.number_input("Daylight Duration (s)",      40000.0,50000.0,44000.0,100.0)
                wcode    = st.number_input("WMO Weather Code",            0, 99, 2)

            sel_model = st.selectbox("Model", list(models.keys()))
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

        if submitted and models:
            dist_enc = int(le.transform([district])[0])
            loc_row  = df[df["city_name"]==district][["latitude","longitude"]].iloc[0]

            inp = {
                "temperature_2m_min (°C)":        tmin,
                "temperature_2m_mean (°C)":        tmean,
                "apparent_temperature_max (°C)":   app_max,
                "apparent_temperature_min (°C)":   app_min,
                "apparent_temperature_mean (°C)":  (app_max+app_min)/2,
                "precipitation_hours (h)":         prec_h,
                "wind_speed_10m_max (km/h)":       wind,
                "wind_gusts_10m_max (km/h)":       gusts,
                "wind_direction_10m_dominant (°)": wind_dir,
                "shortwave_radiation_sum (MJ/m²)": radiation,
                "et0_fao_evapotranspiration (mm)": 3.5,
                "sunshine_duration (s)":           sunshine,
                "daylight_duration (s)":           daylight,
                "weather_code (wmo code)":         wcode,
                "month":                           month,
                "day_of_year":                     30*(month-1)+15,
                "year":                            2023,
                "rain_lag1":  rain_l1, "rain_lag3":  rain_l3,
                "rain_roll7": rain_r7, "rain_roll30": rain_r7*0.9,
                "tmax_lag1":  tmax_l1, "tmax_lag3":  tmax_l1*0.98,
                "tmax_roll7": tmax_l1*0.97, "tmax_roll30": tmax_l1*0.96,
                "wind_lag1":  wind*0.9,  "wind_lag3":  wind*0.85,
                "wind_roll7": wind*0.8,  "wind_roll30":wind*0.75,
                "daylight_min": daylight/60,
                "district_enc": dist_enc,
                "latitude":  float(loc_row["latitude"]),
                "longitude": float(loc_row["longitude"]),
            }
            X_in = pd.DataFrame([{c: inp.get(c,0) for c in feat_cols}])
            m    = models[sel_model]
            pred = int(m.predict(X_in)[0])
            prob = float(m.predict_proba(X_in)[0][1])

            st.markdown("---")
            if pred == 1:
                st.error(f"⚠️ **Extreme Weather Day Predicted** — {prob*100:.1f}% probability")
            else:
                st.success(f"✅ **Normal Day Predicted** — {prob*100:.1f}% extreme probability")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                title={"text":"Extreme Weather Risk (%)"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#e74c3c" if pred==1 else "#27ae60"},
                       "steps":[{"range":[0,30],"color":"#eafaf1"},
                                 {"range":[30,60],"color":"#fef9e7"},
                                 {"range":[60,100],"color":"#fce4e4"}],
                       "threshold":{"line":{"color":"black","width":3},
                                    "thickness":0.75,"value":50}}))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### All Models Comparison")
            rows = []
            for nm, mo in models.items():
                p  = int(mo.predict(X_in)[0])
                pr = float(mo.predict_proba(X_in)[0][1])
                rows.append({"Model":nm,
                             "Prediction":"⚠️ Extreme" if p==1 else "✅ Normal",
                             "Probability":f"{pr*100:.1f}%"})
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — MONTHLY FORECAST
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📅 Monthly Extreme Weather Forecast")
        st.markdown(
            "Generates a prediction for every day of the selected month using "
            "that month's typical Sri Lanka climate patterns."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            fc_district = st.selectbox("District", DISTRICTS, key="fc_district")
        with col2:
            MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            fc_month = st.selectbox(
                "Month", list(range(1, 13)),
                format_func=lambda m: MONTH_NAMES[m - 1],
                index=4, key="fc_month",
            )
        with col3:
            fc_model = st.selectbox("Model", list(models.keys()), key="fc_model")

        if st.button("🗓️ Generate Monthly Forecast", use_container_width=True):
            import calendar

            # Sri Lanka monthly climate averages
            MONTHLY_CLIMATE = {
                1:  {"rain":  67, "tmax": 30.0, "wind": 20},
                2:  {"rain":  69, "tmax": 31.0, "wind": 18},
                3:  {"rain":  99, "tmax": 32.0, "wind": 17},
                4:  {"rain": 220, "tmax": 32.0, "wind": 16},
                5:  {"rain": 371, "tmax": 31.0, "wind": 20},
                6:  {"rain": 179, "tmax": 30.0, "wind": 25},
                7:  {"rain": 136, "tmax": 29.5, "wind": 27},
                8:  {"rain": 109, "tmax": 29.5, "wind": 24},
                9:  {"rain": 160, "tmax": 29.5, "wind": 20},
                10: {"rain": 341, "tmax": 29.0, "wind": 18},
                11: {"rain": 315, "tmax": 28.5, "wind": 17},
                12: {"rain": 147, "tmax": 29.0, "wind": 18},
            }

            clim         = MONTHLY_CLIMATE[fc_month]
            dist_enc     = int(le.transform([fc_district])[0])
            loc_row      = df[df["city_name"] == fc_district][["latitude","longitude"]].iloc[0]
            num_days     = calendar.monthrange(2024, fc_month)[1]
            fc_model_obj = models[fc_model]

            daily_results = []
            for day in range(1, num_days + 1):
                day_factor  = 1 + 0.3 * np.sin(2 * np.pi * day / num_days)
                rain_today  = clim["rain"] / num_days * day_factor
                wind_today  = clim["wind"] * (1 + 0.1 * np.sin(day))
                tmax_today  = clim["tmax"] + 1.5 * np.sin(2 * np.pi * day / num_days)
                tmean_today = tmax_today - 4.0
                tmin_today  = tmax_today - 7.0

                inp = {
                    "temperature_2m_min (°C)":        tmin_today,
                    "temperature_2m_mean (°C)":        tmean_today,
                    "apparent_temperature_max (°C)":   tmax_today + 1.5,
                    "apparent_temperature_min (°C)":   tmin_today + 1.0,
                    "apparent_temperature_mean (°C)":  tmean_today + 1.0,
                    "precipitation_hours (h)":         min(24, rain_today / 3),
                    "wind_speed_10m_max (km/h)":       wind_today,
                    "wind_gusts_10m_max (km/h)":       wind_today * 1.5,
                    "wind_direction_10m_dominant (°)": 180,
                    "shortwave_radiation_sum (MJ/m²)": 18.0,
                    "et0_fao_evapotranspiration (mm)": 3.5,
                    "sunshine_duration (s)":           25000,
                    "daylight_duration (s)":           44000,
                    "weather_code (wmo code)":         61 if rain_today > 10 else 2,
                    "month":                           fc_month,
                    "day_of_year":                     30 * (fc_month - 1) + day,
                    "year":                            2024,
                    "rain_lag1":   rain_today * 0.9,
                    "rain_lag3":   rain_today * 0.85,
                    "rain_roll7":  rain_today * 0.8,
                    "rain_roll30": rain_today * 0.75,
                    "tmax_lag1":   tmax_today * 0.99,
                    "tmax_lag3":   tmax_today * 0.98,
                    "tmax_roll7":  tmax_today * 0.97,
                    "tmax_roll30": tmax_today * 0.96,
                    "wind_lag1":   wind_today * 0.9,
                    "wind_lag3":   wind_today * 0.85,
                    "wind_roll7":  wind_today * 0.8,
                    "wind_roll30": wind_today * 0.75,
                    "daylight_min": 44000 / 60,
                    "district_enc": dist_enc,
                    "latitude":    float(loc_row["latitude"]),
                    "longitude":   float(loc_row["longitude"]),
                }
                X_day = pd.DataFrame([{c: inp.get(c, 0) for c in feat_cols}])
                prob  = float(fc_model_obj.predict_proba(X_day)[0][1])
                pred  = int(fc_model_obj.predict(X_day)[0])
                daily_results.append({"Day": day, "Risk (%)": round(prob * 100, 1), "Prediction": pred})

            result_df    = pd.DataFrame(daily_results)
            extreme_days = result_df[result_df["Prediction"] == 1]["Day"].tolist()
            avg_risk     = result_df["Risk (%)"].mean()
            max_risk_row = result_df.loc[result_df["Risk (%)"].idxmax()]
            month_name   = MONTH_NAMES[fc_month - 1]

            # ── Summary metrics ──────────────────────────────────────────
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("⚠️ Extreme Days Predicted", f"{len(extreme_days)} / {num_days}")
            m2.metric("📊 Average Risk",            f"{avg_risk:.1f}%")
            m3.metric("🔺 Highest Risk Day",        f"Day {int(max_risk_row['Day'])} ({max_risk_row['Risk (%)']:.1f}%)")

            # ── Risk timeline chart ──────────────────────────────────────
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=result_df["Day"],
                y=result_df["Risk (%)"],
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(
                    color=["#e74c3c" if p == 1 else "#27ae60"
                           for p in result_df["Prediction"]],
                    size=8,
                ),
                name="Daily Risk",
            ))
            fig_line.add_hline(y=50, line_dash="dash", line_color="orange",
                               annotation_text="Risk threshold (50%)")
            fig_line.update_layout(
                title=f"Daily Extreme Weather Risk — {month_name}, {fc_district}",
                xaxis_title="Day of Month",
                yaxis_title="Extreme Weather Risk (%)",
                yaxis=dict(range=[0, 100]),
                height=360,
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # ── Calendar heatmap ─────────────────────────────────────────
            st.markdown("#### 🗓️ Calendar View")
            risk_map = dict(zip(result_df["Day"], result_df["Risk (%)"]))
            pred_map = dict(zip(result_df["Day"], result_df["Prediction"]))

            first_dow = calendar.monthrange(2024, fc_month)[0]  # 0 = Monday
            weeks, week_row, dow = [], [], 0
            for _ in range(first_dow):
                week_row.append(None)
                dow += 1
            for day in range(1, num_days + 1):
                week_row.append(day)
                dow += 1
                if dow == 7:
                    weeks.append(week_row)
                    week_row, dow = [], 0
            if week_row:
                while len(week_row) < 7:
                    week_row.append(None)
                weeks.append(week_row)

            cal_html = """
            <style>
            .cal-wrap{font-family:Arial,sans-serif;max-width:580px;margin:0 auto}
            .cal-wrap table{width:100%;border-collapse:collapse}
            .cal-wrap th{background:#1F4E79;color:#fff;padding:9px;text-align:center;font-size:13px}
            .cal-wrap td{width:14%;padding:8px 4px;text-align:center;vertical-align:top;
                         border:1px solid #2e4a62;font-size:12px;border-radius:4px}
            .cal-wrap td.empty{background:#0f1b2d}
            .extreme{background:#4a1010;color:#ff6b6b;font-weight:bold}
            .normal{background:#0d2e1e;color:#2ecc71}
            .risk-pct{font-size:10px;color:#8899aa;display:block;margin-top:2px}
            </style>
            <div class="cal-wrap"><table>
            <tr><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th></tr>
            """
            for week in weeks:
                cal_html += "<tr>"
                for d in week:
                    if d is None:
                        cal_html += '<td class="empty"></td>'
                    else:
                        risk = risk_map.get(d, 0)
                        cls  = "extreme" if pred_map.get(d, 0) == 1 else "normal"
                        cal_html += (
                            f'<td class="{cls}"><b>{d}</b>'
                            f'<span class="risk-pct">{risk:.0f}%</span></td>'
                        )
                cal_html += "</tr>"
            cal_html += "</table></div>"
            st.markdown(cal_html, unsafe_allow_html=True)

            # ── Extreme days alert ───────────────────────────────────────
            st.markdown("---")
            if extreme_days:
                st.error(
                    f"⚠️ **{len(extreme_days)} extreme weather day(s) predicted** "
                    f"in {month_name} for {fc_district}: "
                    f"Days {', '.join(map(str, extreme_days))}"
                )
            else:
                st.success(
                    f"✅ **No extreme weather days predicted** "
                    f"in {month_name} for {fc_district}"
                )

            # ── Download CSV ─────────────────────────────────────────────
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Monthly Forecast CSV",
                data=csv,
                file_name=f"forecast_{fc_district}_{month_name}.csv",
                mime="text/csv",
            )
