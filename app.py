"""
CSY3025 — California Housing Price Predictor
Streamlit GUI: XGBoost vs Linear Regression
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st

# Always resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #3a3d5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #7c6af7; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .price-box {
        background: linear-gradient(135deg, #1a2e1a, #1e3a1e);
        border: 2px solid #4ade80;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .price-value { font-size: 2.8rem; font-weight: bold; color: #4ade80; }
    .price-label { font-size: 1rem; color: #9ca3af; }
    .lr-box {
        background: linear-gradient(135deg, #2e1a1a, #3a1e1e);
        border: 2px solid #f97316;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .lr-value { font-size: 2.8rem; font-weight: bold; color: #f97316; }
    .winner-badge {
        background: #4ade80;
        color: #0f1117;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #e2e8f0;
        margin: 20px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #7c6af7;
    }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading (cached) ────────────────────────────────────
@st.cache_data
def load_and_prepare():
    csv_path = os.path.join(SCRIPT_DIR, "housing.csv")
    df = pd.read_csv(csv_path)

    # Missing values
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

    # Encode categorical
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True, dtype=int)

    # Scale target to $100k units
    df["median_house_value"] = df["median_house_value"] / 100_000

    # Per-household averages
    df["ave_rooms"]  = df["total_rooms"]    / df["households"]
    df["ave_bedrms"] = df["total_bedrooms"] / df["households"]
    df["ave_occup"]  = df["population"]     / df["households"]

    # Outlier clipping
    clip_cols = ["total_rooms","total_bedrooms","population",
                 "households","ave_rooms","ave_bedrms","ave_occup"]
    for col in clip_cols:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # Feature engineering
    df["rooms_per_person"]         = df["ave_rooms"]  / df["ave_occup"]
    df["bedroom_ratio"]            = df["ave_bedrms"] / df["ave_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    df["income_per_room"]          = df["median_income"] / df["ave_rooms"]
    df["dist_sf"]    = np.sqrt((df["latitude"]-37.77)**2 + (df["longitude"]+122.42)**2)
    df["dist_la"]    = np.sqrt((df["latitude"]-34.05)**2 + (df["longitude"]+118.24)**2)
    df["dist_sd"]    = np.sqrt((df["latitude"]-32.72)**2 + (df["longitude"]+117.16)**2)
    df["dist_coast"] = df[["dist_sf","dist_la","dist_sd"]].min(axis=1)
    df["income_x_coast"] = df["median_income"] * (1 / (df["dist_coast"] + 0.1))
    df.dropna(inplace=True)

    return df


@st.cache_resource
def train_models(df_hash):
    df = load_and_prepare()
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Linear Regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)

    # Metrics
    def metrics(y_true, y_pred):
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2)

    lr_m   = metrics(y_test, y_pred_lr)
    xgb_m  = metrics(y_test, y_pred_xgb)

    # Cross-val
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(xgb_model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    cv_mse = -cross_val_score(xgb_model, X, y, cv=kf,
                               scoring="neg_mean_squared_error", n_jobs=-1)

    return (lr, xgb_model, scaler, X_train, X_test, y_train, y_test,
            y_pred_lr, y_pred_xgb, lr_m, xgb_m, cv_r2, cv_mse, X.columns.tolist())


# ── Build Feature Vector from User Input ─────────────────────────────
def build_feature_vector(inputs, feature_cols, df):
    row = {}
    row["longitude"]          = inputs["longitude"]
    row["latitude"]           = inputs["latitude"]
    row["housing_median_age"] = inputs["housing_median_age"]
    row["total_rooms"]        = inputs["total_rooms"]
    row["total_bedrooms"]     = inputs["total_bedrooms"]
    row["population"]         = inputs["population"]
    row["households"]         = inputs["households"]
    row["median_income"]      = inputs["median_income"]

    # One-hot ocean proximity (same columns as training)
    op_cols = [c for c in feature_cols if c.startswith("ocean_proximity_")]
    for col in op_cols:
        row[col] = 0
    prox = inputs["ocean_proximity"]
    # Map display label → encoded column
    prox_map = {
        "INLAND":     "ocean_proximity_INLAND",
        "ISLAND":     "ocean_proximity_ISLAND",
        "NEAR BAY":   "ocean_proximity_NEAR BAY",
        "NEAR OCEAN": "ocean_proximity_NEAR OCEAN",
        "<1H OCEAN":  None   # dropped first, so all zeros = <1H OCEAN
    }
    col_name = prox_map.get(prox)
    if col_name and col_name in row:
        row[col_name] = 1

    # Derived features
    row["ave_rooms"]  = row["total_rooms"]    / max(row["households"], 1)
    row["ave_bedrms"] = row["total_bedrooms"] / max(row["households"], 1)
    row["ave_occup"]  = row["population"]     / max(row["households"], 1)

    row["rooms_per_person"]         = row["ave_rooms"]  / max(row["ave_occup"], 0.01)
    row["bedroom_ratio"]            = row["ave_bedrms"] / max(row["ave_rooms"],  0.01)
    row["population_per_household"] = row["population"] / max(row["households"], 1)
    row["income_per_room"]          = row["median_income"] / max(row["ave_rooms"], 0.01)

    lat, lon = row["latitude"], row["longitude"]
    row["dist_sf"]    = np.sqrt((lat-37.77)**2 + (lon+122.42)**2)
    row["dist_la"]    = np.sqrt((lat-34.05)**2 + (lon+118.24)**2)
    row["dist_sd"]    = np.sqrt((lat-32.72)**2 + (lon+117.16)**2)
    row["dist_coast"] = min(row["dist_sf"], row["dist_la"], row["dist_sd"])
    row["income_x_coast"] = row["median_income"] * (1 / (row["dist_coast"] + 0.1))

    vec = pd.DataFrame([row])[feature_cols]
    return vec


# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════
st.title("🏠 California Housing Price Predictor")
st.caption("CSY3025 · XGBoost vs Linear Regression · Deep Learning Assignment")

# Load data & train
df = load_and_prepare()

with st.spinner("🔄 Training models on California Housing dataset..."):
    (lr, xgb_model, scaler, X_train, X_test, y_train, y_test,
     y_pred_lr, y_pred_xgb, lr_m, xgb_m, cv_r2, cv_mse,
     feature_cols) = train_models(hash(str(df.shape)))

st.success("✅ Models trained and ready!")

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Price", "📊 Model Metrics", "📈 Visualisations", "📋 Dataset Explorer"
])


# ════════════════════════════════════════════════════════════
# TAB 1: PREDICT PRICE
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🔮 Enter House Details to Predict Price</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📍 Location")
        longitude = st.slider("Longitude", -124.35, -114.31, -119.57, step=0.01,
                              help="Western California = lower values")
        latitude  = st.slider("Latitude",   32.54,   41.95,   35.63, step=0.01,
                              help="Northern California = higher values")
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
        )

        st.subheader("🏘️ Block Stats")
        population     = st.number_input("Population",       100, 40000, 1500, step=50)
        households     = st.number_input("Households",        10,  6000,  500, step=10)
        total_rooms    = st.number_input("Total Rooms",       50, 40000, 2500, step=50)
        total_bedrooms = st.number_input("Total Bedrooms",    10,  7000,  500, step=10)

    with col_right:
        st.subheader("📊 Housing Stats")
        housing_median_age = st.slider("Housing Median Age (years)", 1, 52, 28, step=1)
        median_income      = st.slider("Median Income ($10k units)", 0.5, 15.0, 3.87,
                                       step=0.1,
                                       help="E.g., 3.87 = $38,700/yr")

        st.markdown("---")
        st.info(f"""
**📌 Quick Reference**
- **Median Income** 3.87 = ~$38,700/yr
- **Longitude** around -118 = Los Angeles area
- **Latitude** around 37 = San Francisco area
- Target is in **$100k units** — multiply by 100,000 for full USD price
        """)

    # ── Predict ─────────────────────────────────────────────────────
    if st.button("🚀 Predict House Price", use_container_width=True, type="primary"):
        inputs = dict(
            longitude=longitude, latitude=latitude,
            housing_median_age=housing_median_age,
            total_rooms=total_rooms, total_bedrooms=total_bedrooms,
            population=population, households=households,
            median_income=median_income, ocean_proximity=ocean_proximity
        )
        vec = build_feature_vector(inputs, feature_cols, df)
        vec_s = scaler.transform(vec)

        xgb_price = float(xgb_model.predict(vec)[0])
        lr_price  = float(lr.predict(vec_s)[0])

        xgb_usd = xgb_price * 100_000
        lr_usd  = lr_price  * 100_000

        st.markdown("---")
        st.markdown("### 💰 Price Predictions")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
<div class="price-box">
  <div class="price-label">🤖 XGBoost Prediction</div>
  <div class="price-value">${xgb_usd:,.0f}</div>
  <div class="price-label" style="margin-top:8px">({xgb_price:.4f} × $100k units)</div>
</div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
<div class="lr-box">
  <div class="price-label">📉 Linear Regression Prediction</div>
  <div class="lr-value">${lr_usd:,.0f}</div>
  <div class="price-label" style="margin-top:8px">({lr_price:.4f} × $100k units)</div>
</div>""", unsafe_allow_html=True)

        with c3:
            diff = abs(xgb_usd - lr_usd)
            pct  = diff / max(lr_usd, 1) * 100
            better = "XGBoost" if xgb_m["R2"] > lr_m["R2"] else "Linear Reg"
            st.markdown(f"""
<div class="metric-card" style="border-color:#60a5fa">
  <div class="price-label">📐 Difference</div>
  <div style="font-size:2rem;font-weight:bold;color:#60a5fa">${diff:,.0f}</div>
  <div class="price-label">({pct:.1f}% apart)</div>
  <br>
  <div class="price-label">✅ More Accurate Model</div>
  <div style="font-size:1.1rem;font-weight:bold;color:#4ade80">{better}</div>
</div>""", unsafe_allow_html=True)

        # ── Gauge chart ─────────────────────────────────────────
        st.markdown("### 📊 Prediction Gauge")
        fig = go.Figure()
        min_v = df["median_house_value"].min() * 100_000
        max_v = df["median_house_value"].max() * 100_000
        mean_v = df["median_house_value"].mean() * 100_000

        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=xgb_usd,
            delta={"reference": mean_v, "valueformat": ",.0f"},
            title={"text": "XGBoost Predicted Price (USD)", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [min_v, max_v], "tickformat": "$,.0f",
                         "tickcolor": "white"},
                "bar":  {"color": "#7c6af7"},
                "bgcolor": "#1a1d2e",
                "bordercolor": "#444",
                "steps": [
                    {"range": [min_v, mean_v*0.5],  "color": "#1e2030"},
                    {"range": [mean_v*0.5, mean_v],  "color": "#252840"},
                    {"range": [mean_v, max_v],        "color": "#2a2d4e"},
                ],
                "threshold": {
                    "line": {"color": "#f97316", "width": 3},
                    "thickness": 0.75, "value": mean_v
                }
            },
            number={"valueformat": "$,.0f", "font": {"color": "#7c6af7", "size": 28}}
        ))
        fig.update_layout(
            paper_bgcolor="#0f1117", font_color="white", height=300,
            margin=dict(t=60, b=20, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Feature context ──────────────────────────────────────
        with st.expander("🔍 See Engineered Feature Values"):
            vec_display = vec.copy()
            st.dataframe(vec_display.T.rename(columns={0: "Value"}).round(4),
                         use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2: MODEL METRICS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Model Performance Metrics</div>',
                unsafe_allow_html=True)

    # Summary cards
    metrics_list = [
        ("MSE",  lr_m["MSE"],  xgb_m["MSE"],  False),
        ("RMSE", lr_m["RMSE"], xgb_m["RMSE"], False),
        ("MAE",  lr_m["MAE"],  xgb_m["MAE"],  False),
        ("R²",   lr_m["R2"],   xgb_m["R2"],   True),
    ]

    cols = st.columns(4)
    for i, (name, lr_v, xgb_v, higher_better) in enumerate(metrics_list):
        imp = ((xgb_v - lr_v) / abs(lr_v) * 100) if higher_better else \
              ((lr_v - xgb_v) / abs(lr_v) * 100)
        sign = "+" if imp > 0 else ""
        with cols[i]:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{name}</div>
  <div style="margin:8px 0">
    <span style="font-size:1.1rem;color:#f97316">LR: {lr_v:.4f}</span><br>
    <span style="font-size:1.1rem;color:#7c6af7">XGB: {xgb_v:.4f}</span>
  </div>
  <div style="color:#4ade80;font-size:0.9rem;font-weight:bold">{sign}{imp:.1f}% XGBoost</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Cross-validation
    st.subheader("🔁 5-Fold Cross-Validation (XGBoost)")
    cv_cols = st.columns(3)
    with cv_cols[0]:
        st.metric("Mean CV R²",  f"{cv_r2.mean():.4f}", f"±{cv_r2.std():.4f}")
    with cv_cols[1]:
        st.metric("Mean CV MSE", f"{cv_mse.mean():.4f}", f"±{cv_mse.std():.4f}")
    with cv_cols[2]:
        st.metric("CV Folds", "5", "KFold shuffle=True")

    st.markdown("---")

    # Comparison bar chart
    st.subheader("📊 Error Metrics Comparison")
    fig = go.Figure()
    metric_names = ["MSE", "RMSE", "MAE"]
    lr_vals  = [lr_m["MSE"],  lr_m["RMSE"],  lr_m["MAE"]]
    xgb_vals = [xgb_m["MSE"], xgb_m["RMSE"], xgb_m["MAE"]]

    fig.add_trace(go.Bar(name="Linear Regression", x=metric_names, y=lr_vals,
                         marker_color="#f97316", text=[f"{v:.4f}" for v in lr_vals],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="XGBoost", x=metric_names, y=xgb_vals,
                         marker_color="#7c6af7", text=[f"{v:.4f}" for v in xgb_vals],
                         textposition="outside"))
    fig.update_layout(
        barmode="group", title="Error Metrics (lower = better)",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
        font_color="white", legend=dict(bgcolor="#1a1d2e"),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # R2 bar chart
    col_r2a, col_r2b = st.columns(2)
    with col_r2a:
        fig2 = go.Figure(go.Bar(
            x=["Linear Regression", "XGBoost"],
            y=[lr_m["R2"], xgb_m["R2"]],
            marker_color=["#f97316", "#7c6af7"],
            text=[f"{lr_m['R2']:.4f}", f"{xgb_m['R2']:.4f}"],
            textposition="outside", width=0.4
        ))
        fig2.update_layout(
            title="R² Score (higher = better)", yaxis_range=[0, 1.1],
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
            font_color="white", height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_r2b:
        # Radar chart
        categories = ["R² (norm)", "1-MSE (norm)", "1-RMSE (norm)", "1-MAE (norm)"]
        lr_radar  = [lr_m["R2"],
                     1 - lr_m["MSE"]/max(lr_m["MSE"], xgb_m["MSE"]),
                     1 - lr_m["RMSE"]/max(lr_m["RMSE"], xgb_m["RMSE"]),
                     1 - lr_m["MAE"]/max(lr_m["MAE"], xgb_m["MAE"])]
        xgb_radar = [xgb_m["R2"],
                     1 - xgb_m["MSE"]/max(lr_m["MSE"], xgb_m["MSE"]),
                     1 - xgb_m["RMSE"]/max(lr_m["RMSE"], xgb_m["RMSE"]),
                     1 - xgb_m["MAE"]/max(lr_m["MAE"], xgb_m["MAE"])]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(r=lr_radar+[lr_radar[0]], theta=categories+[categories[0]],
                                        fill="toself", name="Linear Regression",
                                        line_color="#f97316", fillcolor="rgba(249,115,22,0.2)"))
        fig3.add_trace(go.Scatterpolar(r=xgb_radar+[xgb_radar[0]], theta=categories+[categories[0]],
                                        fill="toself", name="XGBoost",
                                        line_color="#7c6af7", fillcolor="rgba(124,106,247,0.2)"))
        fig3.update_layout(
            polar=dict(bgcolor="#1a1d2e",
                       radialaxis=dict(visible=True, range=[0,1], color="white"),
                       angularaxis=dict(color="white")),
            paper_bgcolor="#0f1117", font_color="white",
            title="Model Performance Radar", height=350,
            legend=dict(bgcolor="#1a1d2e")
        )
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3: VISUALISATIONS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📈 Model Visualisations</div>',
                unsafe_allow_html=True)

    y_test_arr = y_test.values

    # Predicted vs Actual
    st.subheader("🎯 Predicted vs Actual")
    fig = go.Figure()
    mn, mx = float(y_test_arr.min()), float(y_test_arr.max())
    fig.add_trace(go.Scatter(x=y_test_arr, y=y_pred_lr,  mode="markers",
                             name="Linear Regression", opacity=0.4,
                             marker=dict(color="#f97316", size=4)))
    fig.add_trace(go.Scatter(x=y_test_arr, y=y_pred_xgb, mode="markers",
                             name="XGBoost", opacity=0.4,
                             marker=dict(color="#7c6af7", size=4)))
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             name="Perfect Fit", line=dict(color="red", dash="dash")))
    fig.update_layout(
        xaxis_title="Actual ($100k)", yaxis_title="Predicted ($100k)",
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
        font_color="white", height=450, legend=dict(bgcolor="#1a1d2e")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("📉 Residual Distribution")
        res_lr  = y_test_arr - y_pred_lr
        res_xgb = y_test_arr - y_pred_xgb
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res_lr,  name=f"LR  std={res_lr.std():.3f}",
                                   opacity=0.7, marker_color="#f97316", nbinsx=60))
        fig.add_trace(go.Histogram(x=res_xgb, name=f"XGB std={res_xgb.std():.3f}",
                                   opacity=0.7, marker_color="#7c6af7", nbinsx=60))
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.update_layout(
            barmode="overlay", xaxis_title="Residual", yaxis_title="Count",
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
            font_color="white", height=350, legend=dict(bgcolor="#1a1d2e")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_v2:
        st.subheader("🔍 Residual vs Fitted (XGBoost)")
        fig = go.Figure(go.Scatter(
            x=y_pred_xgb, y=y_test_arr - y_pred_xgb,
            mode="markers", opacity=0.35,
            marker=dict(color="#7c6af7", size=3)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Predicted ($100k)", yaxis_title="Residuals",
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d2e",
            font_color="white", height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("⭐ XGBoost Feature Importance")
    fi_df = pd.DataFrame({"feature": feature_cols,
                          "importance": xgb_model.feature_importances_})
    fi_df = fi_df.sort_values("importance", ascending=True)
    med_imp = fi_df["importance"].median()
    colors = ["#4ade80" if v > med_imp else "#555" for v in fi_df["importance"]]

    fig = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["feature"],
        orientation="h", marker_color=colors
    ))
    fig.update_layout(
        xaxis_title="Importance Score", paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d2e", font_color="white", height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Geographic map
    st.subheader("🗺️ Geographic Distribution of House Prices")
    df_sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        df_sample, x="longitude", y="latitude",
        color="median_house_value",
        color_continuous_scale="Plasma",
        title="Actual Median House Values — California",
        labels={"median_house_value": "Price ($100k)",
                "longitude": "Longitude", "latitude": "Latitude"},
        opacity=0.7, size_max=4,
        template="plotly_dark"
    )
    fig.update_layout(paper_bgcolor="#0f1117", height=500)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4: DATASET EXPLORER
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📋 Dataset Explorer</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", len(feature_cols))
    c3.metric("Train Size", f"{int(len(df)*0.8):,}")
    c4.metric("Test Size",  f"{int(len(df)*0.2):,}")

    st.subheader("📊 Dataset Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.subheader("👀 Sample Rows")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📈 Target Distribution")
    fig = px.histogram(df, x="median_house_value", nbins=60,
                       color_discrete_sequence=["#7c6af7"],
                       labels={"median_house_value": "Median House Value ($100k)"},
                       template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0f1117", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem'>"
    "CSY3025 · Advanced Predictive Modelling · California Housing Dataset"
    "</div>", unsafe_allow_html=True
)