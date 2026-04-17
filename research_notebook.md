# Research Notebook: Cross-Sectional Alpha Factor Combiner

## Quantitative Equity — Machine Learning Alpha Research

### Author: Quantitative Research Desk | Date: 2024-Q4

---

> **Abstract:** This notebook operationalises a production-grade, tree-based alpha factor combination framework for
> cross-sectional equity return prediction. We construct a rich feature space of technical and statistical factors, train
> an XGBoost regressor under a rigorous Purged Time-Series Cross-Validation regime, and leverage SHAP (SHapley Additive
> exPlanations) to decompose the model's predictions into auditable, factor-level attribution. The universe is a curated
> basket of large-cap technology equities sourced from Yahoo Finance.

---

## §0 — Environment & Library Imports

```python
# ── Standard Library ──────────────────────────────────────────────────────────
import sys
import warnings
import logging
from pathlib import Path

# ── Scientific Stack ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ── Project Source Modules ────────────────────────────────────────────────────
sys.path.insert(0, str(Path("..").resolve()))
from src.feature_engineering import FeatureEngineer
from src.ml_pipeline import AlphaModel, PurgedTimeSeriesSplit
from src.explainer import ModelExplainer

# ── External Data ─────────────────────────────────────────────────────────────
import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:.6f}".format)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
```

---

## §1 — Universe Definition & Data Acquisition

We construct a concentrated technology equity universe spanning the largest constituents by market capitalisation. Daily
OHLCV data is sourced from Yahoo Finance for the period 2015–2024, providing approximately nine years of history —
sufficient for multiple market regimes (bull, bear, COVID shock, rate-cycle repricing).

```python
UNIVERSE: dict = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms Inc.",
}

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
TICKERS = list(UNIVERSE.keys())

logging.info(f"Downloading OHLCV data for {TICKERS} from {START_DATE} to {END_DATE}.")

raw_data: dict[str, pd.DataFrame] = {}
for ticker in TICKERS:
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    raw_data[ticker] = df
    logging.info(f"  {ticker}: {len(df):,} rows | {df.index.min().date()} → {df.index.max().date()}")

print(f"\n✓ Downloaded data for {len(raw_data)} equities.")
print(f"  Sample schema ({TICKERS[0]}):\n{raw_data[TICKERS[0]].head(3)}")
```

### §1.1 — Price Series Visualisation

```python
fig, axes = plt.subplots(len(TICKERS), 1, figsize=(14, 18), sharex=True)
fig.suptitle(
    "Adjusted Close Price — Tech Equity Basket (2015–2024)",
    fontsize=15, fontweight="bold", y=1.01
)

PALETTE = sns.color_palette("tab10", len(TICKERS))
for ax, (ticker, color) in zip(axes, zip(TICKERS, PALETTE)):
    close = raw_data[ticker]["Close"]
    ax.plot(close.index, close.values, linewidth=0.9, color=color)
    ax.fill_between(close.index, close.values, alpha=0.08, color=color)
    ax.set_ylabel(f"{ticker}\n(USD)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.xlabel("Date", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_series.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Price series chart saved.")
```

---

## §2 — Feature Engineering

The `FeatureEngineer` class computes a multi-dimensional factor space across four categories:

| Category                        | Factors                                                                                                  |
|---------------------------------|----------------------------------------------------------------------------------------------------------|
| **Momentum**                    | Log-return sums over 5, 10, 21, 63 calendar days                                                         |
| **Mean-Reversion**              | Rolling Z-score of price over 20-day and 60-day windows                                                  |
| **Trend / Oscillator**          | RSI(14), MACD Line, MACD Signal, MACD Histogram                                                          |
| **Volatility / Microstructure** | Annualised realised volatility (5d, 10d, 21d), vol ratio, ATR(14), volume Z-score, dollar-volume Z-score |

All features are computed via vectorised `pandas` operations; no Python-level loops touch the time-series arrays.

```python
FORWARD_DAYS = 5  # Prediction horizon: 1-week forward log-return

fe = FeatureEngineer(price_col="Close", volume_col="Volume")

all_features: list[pd.DataFrame] = []
all_targets: list[pd.Series] = []

for ticker in TICKERS:
    df = raw_data[ticker].copy()

    feat_df = fe.fit_transform(df)
    feat_df.columns = [f"{ticker}_{col}" for col in feat_df.columns]

    target = FeatureEngineer.build_target(
        df["Close"], forward_days=FORWARD_DAYS, log_return=True
    ).rename(ticker)

    all_features.append(feat_df)
    all_targets.append(target)
    logging.info(f"  {ticker}: {feat_df.shape[1]} features generated.")

logging.info(
    f"Feature generation complete. Feature space dimensionality: {all_features[0].shape[1] // len(TICKERS)} per-asset factors.")
```

### §2.1 — Panel Construction (Stacked Cross-Section)

```python
panel_features = pd.concat(all_features, axis=0).sort_index()
panel_targets = pd.concat(all_targets, axis=0).sort_index()

combined = pd.concat([panel_features, panel_targets.rename("target")], axis=1).dropna()

X_raw = combined.drop(columns=["target"])
y_raw = combined["target"]

print(f"✓ Panel constructed.")
print(f"  Observations : {len(combined):,}")
print(f"  Features     : {X_raw.shape[1]}")
print(f"  Target       : {FORWARD_DAYS}d forward log-return")
print(f"  Date Range   : {combined.index.min().date()} → {combined.index.max().date()}")
print(f"\n  Target Distribution:")
print(y_raw.describe().to_string())
```

### §2.2 — Feature Correlation Heatmap

```python
sample_features = raw_data[TICKERS[0]].copy()
feat_single = FeatureEngineer().fit_transform(sample_features)

corr_matrix = feat_single.corr(method="spearman")

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.4,
    ax=ax,
)
ax.set_title(
    f"Spearman Rank Correlation — Engineered Feature Matrix ({TICKERS[0]})",
    fontsize=13, fontweight="bold", pad=12
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Feature correlation heatmap saved.")
```

---

## §3 — Model Training with Purged Time-Series Cross-Validation

### §3.1 — Cross-Validation Architecture

The central methodological challenge in financial ML is preventing **look-ahead bias** — the inadvertent leakage of
future information into the training set. Standard `k-fold` CV shuffles observations randomly, which in a time-series
context allows the model to train on future data while being tested on the past, producing grotesquely optimistic
out-of-sample estimates.

We address this through `PurgedTimeSeriesSplit`, which enforces:

1. **Temporal ordering**: All training observations precede all test observations.
2. **Purge gap**: A mandatory buffer of `purge_gap` periods is removed at the boundary of the train/test split,
   eliminating any overlap between the feature window (look-back) and the target horizon (look-forward).
3. **Embargo**: An additional `embargo_pct` of the sample is held out after the test set to prevent the model from
   learning to exploit serial correlation.

```python
alpha_model = AlphaModel(
    xgb_params={
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.015,
        "subsample": 0.75,
        "colsample_bytree": 0.65,
        "min_child_weight": 12,
        "gamma": 0.15,
        "reg_alpha": 0.08,
        "reg_lambda": 1.5,
    },
    n_splits=5,
    purge_gap=FORWARD_DAYS + 1,  # Must be >= prediction horizon
    embargo_pct=0.02,
    early_stopping_rounds=50,
)

logging.info("Initiating training with Purged Time-Series CV...")
training_result = alpha_model.train(X_raw, y_raw)
logging.info("Training complete.")
```

### §3.2 — Cross-Validation Results

```python
print("=" * 65)
print("  PURGED TIME-SERIES CV — OUT-OF-SAMPLE PERFORMANCE SUMMARY")
print("=" * 65)
print(
    f"  {'Fold':<8} {'Train Start':<14} {'Train End':<14} {'Test Start':<14} {'Test End':<14} {'OOS RMSE':>10} {'OOS IC (ρ)':>12}")
print("-" * 95)

for fold in training_result.cv_folds:
    print(
        f"  {fold.fold_idx:<8} "
        f"{str(fold.train_period[0])[:10]:<14} "
        f"{str(fold.train_period[1])[:10]:<14} "
        f"{str(fold.test_period[0])[:10]:<14} "
        f"{str(fold.test_period[1])[:10]:<14} "
        f"{fold.oos_rmse:>10.6f} "
        f"{fold.oos_ic:>12.4f}"
    )

print("-" * 95)
print(
    f"  {'MEAN':<8} {'':14} {'':14} {'':14} {'':14} {training_result.mean_oos_rmse:>10.6f} {training_result.mean_oos_ic:>12.4f}")
print("=" * 65)
print(f"\n  Final Model — Best Iteration : {training_result.best_iteration}")
print(f"  Feature Dimensionality       : {len(training_result.feature_names)}")
```

### §3.3 — OOS IC Distribution Plot

```python
ic_values = [f.oos_ic for f in training_result.cv_folds]
rmse_values = [f.oos_rmse for f in training_result.cv_folds]
fold_labels = [f"Fold {f.fold_idx}" for f in training_result.cv_folds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# IC Bar Chart
colors_ic = ["#2196F3" if ic > 0 else "#F44336" for ic in ic_values]
bars = ax1.bar(fold_labels, ic_values, color=colors_ic, edgecolor="white", linewidth=0.8)
ax1.axhline(np.mean(ic_values), color="black", linestyle="--", linewidth=1.2,
            label=f"Mean IC = {np.mean(ic_values):.4f}")
ax1.axhline(0, color="grey", linestyle="-", linewidth=0.6, alpha=0.5)
ax1.set_title("OOS Spearman IC by Fold", fontsize=12, fontweight="bold")
ax1.set_ylabel("Information Coefficient (Spearman ρ)")
ax1.legend(fontsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# RMSE Bar Chart
ax2.bar(fold_labels, rmse_values, color="#FF9800", edgecolor="white", linewidth=0.8)
ax2.axhline(np.mean(rmse_values), color="black", linestyle="--", linewidth=1.2,
            label=f"Mean RMSE = {np.mean(rmse_values):.6f}")
ax2.set_title("OOS RMSE by Fold", fontsize=12, fontweight="bold")
ax2.set_ylabel("Root Mean Squared Error")
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.suptitle("Purged Time-Series CV — Out-of-Sample Performance", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_performance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ CV performance chart saved.")
```

---

## §4 — SHAP Model Explainability

SHAP (SHapley Additive exPlanations) provides a game-theoretic framework for decomposing any model prediction into
additive contributions from each input feature. For tree ensembles, the `TreeExplainer` computes exact Shapley values in
**O(TLD²)** time (T = trees, L = leaves, D = max depth), without sampling approximations. This is categorically superior
to post-hoc linear proxies (e.g., LIME) for non-linear models.

```python
X_for_shap, y_for_shap = alpha_model._align_and_drop_na(X_raw, y_raw)
X_scaled_for_shap = pd.DataFrame(
    training_result.scaler.transform(X_for_shap),
    index=X_for_shap.index,
    columns=X_for_shap.columns,
)

explainer = ModelExplainer(
    model=training_result.model,
    feature_names=training_result.feature_names,
    output_dir=OUTPUT_DIR / "shap",
)

BACKGROUND_SAMPLE_SIZE = 500
background_idx = np.random.choice(len(X_scaled_for_shap), BACKGROUND_SAMPLE_SIZE, replace=False)
X_background = X_scaled_for_shap.iloc[background_idx]

logging.info("Fitting SHAP TreeExplainer...")
explainer.fit(X_background)
logging.info("Computing SHAP values over full panel...")
shap_values = explainer.explain(X_scaled_for_shap, check_additivity=False)
logging.info(f"✓ SHAP values computed. Shape: {shap_values.shape}")
```

### §4.1 — SHAP Beeswarm Summary Plot

The beeswarm plot displays every individual observation's SHAP value for each feature. The x-axis encodes **directional
attribution** (positive = prediction-increasing), while the colour encodes the **feature value** (red = high, blue =
low). This jointly communicates feature importance, directionality, and non-linearity in a single dense visualisation.

```python
summary_path = explainer.plot_summary(
    X=X_scaled_for_shap,
    plot_type="dot",
    filename="shap_beeswarm_summary.png",
)
print(f"✓ SHAP beeswarm summary saved → {summary_path}")

from IPython.display import Image

Image(str(summary_path))
```

### §4.2 — SHAP Bar Chart (Global Feature Importance)

```python
bar_path = explainer.plot_bar_summary(filename="shap_bar_summary.png")
print(f"✓ SHAP bar summary saved → {bar_path}")
Image(str(bar_path))
```

### §4.3 — SHAP vs. XGBoost Gain: Importance Comparison

A critical practitioner insight: XGBoost's native `gain` importance is a **model-centric** metric (how much the
objective function improves at each split), while SHAP is a **prediction-centric** metric (how much each feature shifts
the final output). The two frequently diverge for correlated features, where gain may inflate the importance of a proxy
variable.

```python
gain_importance = alpha_model.get_feature_importance(importance_type="gain")

comparison_path = explainer.plot_feature_importance_comparison(
    xgb_gain_importance=gain_importance,
    filename="importance_comparison.png",
)
print(f"✓ Importance comparison chart saved → {comparison_path}")
Image(str(comparison_path))
```

### §4.4 — Top Feature SHAP Dependence Plots

Dependence plots reveal the functional relationship between a single feature and its SHAP contribution, with the scatter
point colour encoding the value of the most-interacting feature (selected automatically via maximum absolute
covariance).

```python
shap_mean = explainer.mean_absolute_shap()
top_features = shap_mean.head(4).index.tolist()

print(f"Top 4 features by Mean |SHAP|:")
for i, feat in enumerate(top_features, 1):
    print(f"  {i}. {feat}  (mean |φ| = {shap_mean[feat]:.6f})")

for feat in top_features:
    dep_path = explainer.plot_dependence(
        feature=feat,
        interaction_feature="auto",
        X=X_scaled_for_shap,
        filename=f"shap_dep_{feat[:30].replace('/', '_')}.png",
    )
    print(f"  ✓ Dependence plot saved → {dep_path}")
    Image(str(dep_path))
```

### §4.5 — SHAP Waterfall Plot (Single Observation)

The waterfall plot deconstructs one specific prediction into an ordered sum of individual SHAP contributions, starting
from the model's global expected value `E[f(X)]`. This is the primary tool for **trade-level explainability** —
justifying a position to a risk manager or compliance officer.

```python
waterfall_path = explainer.plot_waterfall(sample_idx=0, filename="shap_waterfall_obs0.png")
print(f"✓ Waterfall plot saved → {waterfall_path}")
Image(str(waterfall_path))
```

---

## §5 — Alpha Score Generation & Backtesting Diagnostic

```python
alpha_scores = alpha_model.predict(X_scaled_for_shap)

score_df = pd.concat([alpha_scores, y_for_shap.rename("realised_return")], axis=1)
score_df["quintile"] = pd.qcut(
    score_df["alpha_score"].rank(method="first"),
    q=5,
    labels=["Q1\n(Short)", "Q2", "Q3", "Q4", "Q5\n(Long)"],
)

quintile_returns = score_df.groupby("quintile", observed=True)["realised_return"].mean()
spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

quintile_returns.plot(
    kind="bar",
    ax=axes[0],
    color=sns.color_palette("RdYlGn", 5),
    edgecolor="white",
    linewidth=0.8,
)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title(
    f"Mean {FORWARD_DAYS}d Forward Return by Alpha Quintile\n(Q5–Q1 Spread = {spread:.4%})",
    fontsize=12, fontweight="bold"
)
axes[0].set_ylabel("Mean Forward Log-Return")
axes[0].set_xlabel("Alpha Score Quintile")
axes[0].tick_params(axis="x", rotation=0)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].scatter(
    score_df["alpha_score"].sample(2000, random_state=42),
    score_df.loc[score_df["alpha_score"].sample(2000, random_state=42).index, "realised_return"],
    alpha=0.25, s=8, color="#1565C0", edgecolors="none"
)
z = np.polyfit(score_df["alpha_score"], score_df["realised_return"], 1)
p = np.poly1d(z)
x_line = np.linspace(score_df["alpha_score"].min(), score_df["alpha_score"].max(), 200)
axes[1].plot(x_line, p(x_line), color="#E53935", linewidth=1.5, label="OLS Fit")
axes[1].set_xlabel("Predicted Alpha Score")
axes[1].set_ylabel(f"Realised {FORWARD_DAYS}d Log-Return")
axes[1].set_title("Predicted Alpha vs. Realised Return", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Alpha Factor Combiner — Predictive Validation", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "alpha_validation.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Alpha validation chart saved.")
print(f"\n  Q5–Q1 Spread : {spread:.4%}")
print(f"  Full-Sample Spearman IC : {score_df['alpha_score'].corr(score_df['realised_return'], method='spearman'):.4f}")
```

---

## §6 — Summary Statistics & SHAP Attribution Table

```python
shap_df = explainer.get_shap_dataframe(index=X_scaled_for_shap.index)
mean_shap = explainer.mean_absolute_shap()

summary_table = pd.DataFrame({
    "Mean |SHAP|": mean_shap,
    "SHAP Rank": mean_shap.rank(ascending=False).astype(int),
    "XGB Gain": gain_importance.reindex(mean_shap.index).fillna(0.0),
    "SHAP Std Dev": shap_df.abs().std(),
    "SHAP 95th Pctile": shap_df.abs().quantile(0.95),
}).sort_values("Mean |SHAP|", ascending=False)

print("\n" + "=" * 75)
print("  GLOBAL SHAP ATTRIBUTION — TOP 20 FEATURES")
print("=" * 75)
print(summary_table.head(20).to_string(float_format="{:.6f}".format))
```

---

## §7 — Artifacts Summary

```python
from pathlib import Path

print("\n" + "=" * 55)
print("  GENERATED RESEARCH ARTIFACTS")
print("=" * 55)

all_artifacts = sorted(OUTPUT_DIR.rglob("*.png"))
for path in all_artifacts:
    size_kb = path.stat().st_size / 1024
    print(f"  [{size_kb:6.1f} KB]  {path.relative_to(OUTPUT_DIR.parent)}")

print("\n" + "=" * 55)
print("  END OF RESEARCH NOTEBOOK")
print("=" * 55)
```
