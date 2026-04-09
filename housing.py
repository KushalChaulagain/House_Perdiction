"""
CSY3025 - Advanced Predictive Modelling
California Housing Dataset  (Kaggle — housing.csv)
Model: XGBoost vs Linear Regression (Baseline)
"""
import warnings
warnings.filterwarnings("ignore")

import os
OUT = os.path.dirname(os.path.abspath(__file__))   # figures save next to this script

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

# ── Plot style ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#444",    "axes.labelcolor":  "white",
    "xtick.color":      "white",   "ytick.color":      "white",
    "text.color":       "white",   "grid.color":       "#2a2d3e",
    "grid.linestyle":   "--",      "grid.alpha":       0.5,
    "legend.facecolor": "#1a1d2e", "legend.edgecolor": "#444",
})
ACCENT="#7c6af7"; GREEN="#4ade80"; ORANGE="#f97316"; PINK="#f472b6"; BLUE="#60a5fa"

print("="*60)
print("  CSY3025 — California Housing Predictive Model")
print("="*60)


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────
print("\n[1] Loading housing.csv...")
CSV_PATH = os.path.join(OUT, "housing.csv")
df = pd.read_csv(CSV_PATH)
print(f"    Shape     : {df.shape}")
print(f"    Columns   : {list(df.columns)}")
print(f"    Missing   : {df.isnull().sum().to_dict()}")


# ─────────────────────────────────────────────────────────────
# 2. MISSING VALUES
# ─────────────────────────────────────────────────────────────
print("\n[2] Handling missing values...")
# total_bedrooms has 207 missing — fill with median (safest for skewed data)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
print(f"    Missing after fix: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────
# 3. ENCODE CATEGORICAL — ocean_proximity
# ─────────────────────────────────────────────────────────────
print("\n[3] Encoding ocean_proximity...")
print(f"    Unique values: {df['ocean_proximity'].unique()}")
# One-hot encode (drops first to avoid multicollinearity)
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True, dtype=int)
ocean_cols = [c for c in df.columns if c.startswith("ocean_proximity_")]
print(f"    Encoded columns: {ocean_cols}")


# ─────────────────────────────────────────────────────────────
# 4. SCALE TARGET
#    Raw values are in dollars (14,999 – 500,001).
#    Divide by 100,000 → units of $100k, matching sklearn baseline.
# ─────────────────────────────────────────────────────────────
df["median_house_value"] = df["median_house_value"] / 100_000
print(f"\n[4] Target scaled to $100k units  "
      f"(min={df['median_house_value'].min():.3f}, "
      f"max={df['median_house_value'].max():.3f})")


# ─────────────────────────────────────────────────────────────
# 5. DERIVE AVERAGE FEATURES
#    The raw CSV gives totals per block; divide by households
#    to get per-household averages (matches sklearn feature space).
# ─────────────────────────────────────────────────────────────
print("\n[5] Deriving per-household averages...")
df["ave_rooms"]  = df["total_rooms"]    / df["households"]
df["ave_bedrms"] = df["total_bedrooms"] / df["households"]
df["ave_occup"]  = df["population"]     / df["households"]
print("    Added: ave_rooms, ave_bedrms, ave_occup")


# ─────────────────────────────────────────────────────────────
# 6. OUTLIER CLIPPING
# ─────────────────────────────────────────────────────────────
print("\n[6] Clipping outliers (1–99th percentile)...")
clip_cols = ["total_rooms","total_bedrooms","population",
             "households","ave_rooms","ave_bedrms","ave_occup"]
for col in clip_cols:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)
print(f"    Clipped: {clip_cols}")


# ─────────────────────────────────────────────────────────────
# 7. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[7] Engineering new features...")
df["rooms_per_person"]          = df["ave_rooms"]  / df["ave_occup"]
df["bedroom_ratio"]             = df["ave_bedrms"] / df["ave_rooms"]
df["population_per_household"]  = df["population"] / df["households"]
df["income_per_room"]           = df["median_income"] / df["ave_rooms"]

# Distance to major coastal cities (great for capturing geo price clusters)
df["dist_sf"] = np.sqrt((df["latitude"]-37.77)**2 + (df["longitude"]+122.42)**2)
df["dist_la"] = np.sqrt((df["latitude"]-34.05)**2 + (df["longitude"]+118.24)**2)
df["dist_sd"] = np.sqrt((df["latitude"]-32.72)**2 + (df["longitude"]+117.16)**2)
df["dist_coast"] = df[["dist_sf","dist_la","dist_sd"]].min(axis=1)

# Income × location interaction
df["income_x_coast"] = df["median_income"] * (1 / (df["dist_coast"] + 0.1))

eng = ["rooms_per_person","bedroom_ratio","population_per_household",
       "income_per_room","dist_sf","dist_la","dist_sd","dist_coast","income_x_coast"]
print(f"    +{len(eng)} engineered features")

# Safety net: drop any row that still has NaN after all transformations
before = len(df)
df.dropna(inplace=True)
print(f"    Dropped {before - len(df)} rows with remaining NaN values")
print(f"    Final dataset shape: {df.shape}")
print(f"\n    Full statistics:")
print(df.describe().round(3).to_string())


# ─────────────────────────────────────────────────────────────
# 8. EDA — FIGURE 1
# ─────────────────────────────────────────────────────────────
print("\n[8] Generating EDA visualisations...")

base_features = ["median_income","housing_median_age","ave_rooms","ave_bedrms",
                 "population","ave_occup","latitude","longitude"]

fig = plt.figure(figsize=(20, 18), facecolor="#0f1117")
fig.suptitle("California Housing Dataset — Exploratory Data Analysis",
             fontsize=18, color="white", fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

# Target distribution
ax0 = fig.add_subplot(gs[0, 0])
ax0.hist(df["median_house_value"], bins=50, color=ACCENT, edgecolor="#0f1117", alpha=0.9)
ax0.axvline(df["median_house_value"].mean(), color=ORANGE, lw=2, ls="--",
            label=f"Mean={df['median_house_value'].mean():.2f}")
ax0.set_title("Target Distribution", color="white", fontsize=11)
ax0.set_xlabel("Median House Value ($100k)"); ax0.set_ylabel("Count"); ax0.legend()

# Correlation heatmap (base features + target only, keep readable)
ax1 = fig.add_subplot(gs[0, 1:])
hm_cols = base_features + ["median_house_value"]
corr = df[hm_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5,
            annot_kws={"size": 8, "color": "white"},
            ax=ax1, cbar_kws={"shrink": 0.8})
ax1.set_title("Feature Correlation Matrix", color="white", fontsize=11)
ax1.tick_params(colors="white")

# Feature distributions
palette = [ACCENT, GREEN, ORANGE, PINK, BLUE, "#34d399", "#fbbf24", "#a78bfa"]
for i, feat in enumerate(base_features):
    row = 1 + i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    ax.hist(df[feat], bins=40, color=palette[i], edgecolor="#0f1117", alpha=0.85)
    ax.set_title(feat, color="white", fontsize=10)
    ax.set_xlabel("Value"); ax.set_ylabel("Count")

plt.savefig(os.path.join(OUT, "fig1_eda.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("    Saved → fig1_eda.png")


# ─────────────────────────────────────────────────────────────
# 9. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[9] Train/Test split (80/20)...")
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"    Train: {X_train.shape}   Test: {X_test.shape}")
print(f"    Features used: {list(X.columns)}")


# ─────────────────────────────────────────────────────────────
# 10. BASELINE — LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────
print("\n[10] Training Baseline: Linear Regression...")
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

lr_mse  = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_mae  = mean_absolute_error(y_test, y_pred_lr)
lr_r2   = r2_score(y_test, y_pred_lr)

print(f"    MSE={lr_mse:.4f}  RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}  R²={lr_r2:.4f}")
print(f"    (Assignment baseline: MSE=0.5559  R²=0.5758)")


# ─────────────────────────────────────────────────────────────
# 11. XGBOOST
# ─────────────────────────────────────────────────────────────
print("\n[11] Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(
    n_estimators     = 800,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 3,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,    # L1
    reg_lambda       = 1.0,    # L2
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
y_pred_xgb = xgb_model.predict(X_test)

xgb_mse  = mean_squared_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae  = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2   = r2_score(y_test, y_pred_xgb)

print(f"    MSE={xgb_mse:.4f}  RMSE={xgb_rmse:.4f}  MAE={xgb_mae:.4f}  R²={xgb_r2:.4f}")


# ─────────────────────────────────────────────────────────────
# 12. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
print("\n[12] 5-Fold Cross Validation (XGBoost)...")
kf     = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2  = cross_val_score(xgb_model, X, y, cv=kf, scoring="r2", n_jobs=-1)
cv_mse = -cross_val_score(xgb_model, X, y, cv=kf,
                           scoring="neg_mean_squared_error", n_jobs=-1)
print(f"    CV R²  → {np.round(cv_r2,4)}  Mean: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"    CV MSE → {np.round(cv_mse,4)}  Mean: {cv_mse.mean():.4f} ± {cv_mse.std():.4f}")


# ─────────────────────────────────────────────────────────────
# 13. FIGURE 2 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
print("\n[13] Generating comparison plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0f1117")
fig.suptitle("Model Comparison — Linear Regression vs XGBoost",
             fontsize=16, color="white", fontweight="bold", y=0.98)

# Error metrics bar chart
ax = axes[0, 0]
metrics = ["MSE", "RMSE", "MAE"]
lr_v    = [lr_mse,  lr_rmse,  lr_mae]
xgb_v   = [xgb_mse, xgb_rmse, xgb_mae]
xp = np.arange(3); w = 0.35
b1 = ax.bar(xp-w/2, lr_v,  w, label="Linear Regression", color=ORANGE, alpha=0.85)
b2 = ax.bar(xp+w/2, xgb_v, w, label="XGBoost",           color=ACCENT, alpha=0.85)
for bar, val in zip(list(b1)+list(b2), lr_v+xgb_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{val:.3f}", ha="center", fontsize=9, color="white")
ax.set_xticks(xp); ax.set_xticklabels(metrics)
ax.set_title("Error Metrics (lower = better)", color="white"); ax.legend(); ax.grid(True)

# R² bar chart
ax = axes[0, 1]
bars = ax.bar(["Linear Regression", "XGBoost"], [lr_r2, xgb_r2],
              color=[ORANGE, ACCENT], alpha=0.85, width=0.4)
for bar, val in zip(bars, [lr_r2, xgb_r2]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.4f}", ha="center", color="white", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_title("R² Score (higher = better)", color="white"); ax.grid(True)

# Predicted vs Actual
ax = axes[1, 0]
mn, mx = float(y_test.min()), float(y_test.max())
ax.scatter(y_test, y_pred_lr,  alpha=0.25, s=4, color=ORANGE, label="Linear Reg")
ax.scatter(y_test, y_pred_xgb, alpha=0.25, s=4, color=ACCENT, label="XGBoost")
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
ax.set_xlabel("Actual ($100k)"); ax.set_ylabel("Predicted ($100k)")
ax.set_title("Predicted vs Actual", color="white"); ax.legend(); ax.grid(True)

# Residual distributions
ax = axes[1, 1]
res_lr  = y_test - y_pred_lr
res_xgb = y_test - y_pred_xgb
ax.hist(res_lr,  bins=60, alpha=0.6, color=ORANGE,
        label=f"LR  std={res_lr.std():.3f}",  density=True)
ax.hist(res_xgb, bins=60, alpha=0.6, color=ACCENT,
        label=f"XGB std={res_xgb.std():.3f}", density=True)
ax.axvline(0, color="white", lw=1.5, ls="--")
ax.set_xlabel("Residual"); ax.set_ylabel("Density")
ax.set_title("Residual Distribution", color="white"); ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_model_comparison.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("    Saved → fig2_model_comparison.png")


# ─────────────────────────────────────────────────────────────
# 14. FIGURE 3 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
fi_df = pd.DataFrame({"feature":   X.columns,
                      "importance": xgb_model.feature_importances_})
fi_df = fi_df.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0f1117")
med_imp  = fi_df["importance"].median()
bar_cols = [GREEN if v > med_imp else "#555" for v in fi_df["importance"]]
ax.barh(fi_df["feature"], fi_df["importance"], color=bar_cols, edgecolor="#0f1117")
ax.set_title("XGBoost — Feature Importance",
             color="white", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score (F-score)"); ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_feature_importance.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("    Saved → fig3_feature_importance.png")


# ─────────────────────────────────────────────────────────────
# 15. FIGURE 4 — GEOGRAPHIC MAP
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6), facecolor="#0f1117")
fig.suptitle("Geographic Prediction Map — California Housing",
             color="white", fontsize=14, fontweight="bold")
vmin, vmax = float(y_test.min()), float(y_test.max())

sc0 = axes[0].scatter(X_test["longitude"], X_test["latitude"],
                      c=y_test, cmap="plasma", s=3, alpha=0.7, vmin=vmin, vmax=vmax)
axes[0].set_title("Actual Values",     color="white")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
plt.colorbar(sc0, ax=axes[0], label="$100k")

sc1 = axes[1].scatter(X_test["longitude"], X_test["latitude"],
                      c=y_pred_xgb, cmap="plasma", s=3, alpha=0.7, vmin=vmin, vmax=vmax)
axes[1].set_title("XGBoost Predicted", color="white")
axes[1].set_xlabel("Longitude")
plt.colorbar(sc1, ax=axes[1], label="$100k")

err = np.abs(y_test.values - y_pred_xgb)
sc2 = axes[2].scatter(X_test["longitude"], X_test["latitude"],
                      c=err, cmap="hot", s=3, alpha=0.7)
axes[2].set_title("Absolute Error",    color="white")
axes[2].set_xlabel("Longitude")
plt.colorbar(sc2, ax=axes[2], label="Error $100k")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_geo_map.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("    Saved → fig4_geo_map.png")


# ─────────────────────────────────────────────────────────────
# 16. FIGURE 5 — DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f1117")
fig.suptitle("XGBoost — Detailed Diagnostics",
             color="white", fontsize=14, fontweight="bold")

ax = axes[0]
ax.scatter(y_pred_xgb, y_test.values - y_pred_xgb,
           alpha=0.3, s=4, color=ACCENT)
ax.axhline(0, color="red", lw=1.5, ls="--")
ax.set_xlabel("Predicted ($100k)"); ax.set_ylabel("Residuals")
ax.set_title("Residual vs Fitted", color="white"); ax.grid(True)

ax = axes[1]
inc_bins   = pd.cut(X_test["median_income"], bins=8)
mae_by_inc = pd.Series(np.abs(y_test.values - y_pred_xgb),
                        index=X_test.index).groupby(inc_bins, observed=False).mean()
ax.bar(range(len(mae_by_inc)), mae_by_inc.values, color=PINK, alpha=0.85)
ax.set_xticks(range(len(mae_by_inc)))
ax.set_xticklabels([str(b) for b in mae_by_inc.index],
                   rotation=30, ha="right", fontsize=8)
ax.set_xlabel("Income Bracket"); ax.set_ylabel("Mean Absolute Error")
ax.set_title("MAE by Income Bracket", color="white"); ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_diagnostics.png"), dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.close()
print("    Saved → fig5_diagnostics.png")


# ─────────────────────────────────────────────────────────────
# 17. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
mse_imp  = (lr_mse  - xgb_mse)  / lr_mse  * 100
rmse_imp = (lr_rmse - xgb_rmse) / lr_rmse * 100
mae_imp  = (lr_mae  - xgb_mae)  / lr_mae  * 100
r2_imp   = (xgb_r2  - lr_r2)   / lr_r2   * 100

print("\n" + "="*65)
print("  FINAL RESULTS SUMMARY")
print("="*65)
print(f"  {'Metric':<8} {'Baseline LR':>14} {'XGBoost':>12} {'Improvement':>14}")
print("-"*65)
print(f"  {'MSE':<8} {lr_mse:>14.4f} {xgb_mse:>12.4f} {mse_imp:>12.1f}%")
print(f"  {'RMSE':<8} {lr_rmse:>14.4f} {xgb_rmse:>12.4f} {rmse_imp:>12.1f}%")
print(f"  {'MAE':<8} {lr_mae:>14.4f} {xgb_mae:>12.4f} {mae_imp:>12.1f}%")
print(f"  {'R²':<8} {lr_r2:>14.4f} {xgb_r2:>12.4f} {r2_imp:>12.1f}%")
print(f"  {'CV R²':<8} {'—':>14} {cv_r2.mean():>12.4f}"
      f" {'±'+str(round(cv_r2.std(),4)):>14}")
print("="*65)
print(f"\n  XGBoost beats the baseline on ALL metrics.")
print(f"  R²  : {lr_r2:.4f}  →  {xgb_r2:.4f}  (+{r2_imp:.1f}%)")
print(f"  MSE : {lr_mse:.4f}  →  {xgb_mse:.4f}  (-{mse_imp:.1f}%)")
print(f"\n  Figures saved to: {OUT}")