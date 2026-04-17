# Cross-Sectional Alpha Factor Combiner
## A Production-Grade ML Pipeline for Quantitative Equity Research

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost 2.x](https://img.shields.io/badge/XGBoost-2.x-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-green.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Theoretical Foundations](#3-theoretical-foundations)
   - 3.1 [Why Tree-Based Models Outperform Deep Learning on Tabular Financial Data](#31-why-tree-based-models-outperform-deep-learning-on-tabular-financial-data)
   - 3.2 [SHAP: Game-Theoretic Feature Attribution for Ensemble Models](#32-shap-game-theoretic-feature-attribution-for-ensemble-models)
   - 3.3 [Purged Time-Series Cross-Validation: Rigorous OOS Estimation Under Temporal Dependence](#33-purged-time-series-cross-validation-rigorous-oos-estimation-under-temporal-dependence)
4. [Feature Engineering Methodology](#4-feature-engineering-methodology)
5. [Installation & Usage](#5-installation--usage)
6. [Configuration Reference](#6-configuration-reference)
7. [Empirical Considerations & Known Limitations](#7-empirical-considerations--known-limitations)
8. [References](#8-references)

---

## 1. Project Overview

This repository implements a **Cross-Sectional Alpha Factor Combiner** — a systematic, data-driven approach to combining technical and statistical equity factors into a single, predictive alpha signal. The pipeline ingests daily OHLCV (Open, High, Low, Close, Volume) price data, engineers a multi-dimensional feature space, and trains an `XGBRegressor` to predict forward cross-sectional equity returns.

The framework is designed around three non-negotiable principles that distinguish production quantitative research from academic experimentation:

1. **Temporal Integrity**: All cross-validation, hyperparameter search, and performance reporting respects the causal arrow of time. No future information ever contaminates the training set.
2. **Interpretability by Design**: Every model prediction is decomposable into auditable, factor-level SHAP attributions. Black-box performance is insufficient for risk management and regulatory compliance.
3. **Vectorised Computation**: All feature engineering operations are expressed as vectorised `pandas`/`numpy` transformations. No Python-level loops iterate over time-series rows.

---

## 2. Repository Structure

```
alpha-factor-combiner/
│
├── requirements.txt            # Pinned dependency manifest
│
├── src/
│   ├── feature_engineering.py  # FeatureEngineer: factor construction
│   ├── ml_pipeline.py          # AlphaModel + PurgedTimeSeriesSplit
│   └── explainer.py            # ModelExplainer: SHAP plots & attribution
│
├── research_notebook.md        # Full research workflow (Jupyter-compatible)
├── outputs/                    # Generated charts and SHAP artefacts
└── README.md                   # This document
```

---

## 3. Theoretical Foundations

### 3.1 Why Tree-Based Models Outperform Deep Learning on Tabular Financial Data

The application of deep learning to tabular, cross-sectional financial data is a common and frequently costly mistake. The empirical and theoretical literature has converged on a clear conclusion: for structured, moderate-dimensional tabular datasets, gradient-boosted decision trees (GBDTs) — and specifically XGBoost, LightGBM, and CatBoost — consistently match or surpass deep neural networks. The following arguments underpin this finding.

#### 3.1.1 The Inductive Bias Problem

Deep neural networks are extraordinarily powerful function approximators, but this universality is a double-edged sword. Their inductive bias — the implicit assumptions baked into their architecture — is tuned for **spatially and temporally correlated, high-dimensional, raw perceptual data** (images, audio, text). Standard MLPs have no such structural prior; they must learn from scratch that the relationship between a 21-day volatility estimate and a 5-day volatility estimate is more informative than random cross-feature interactions.

Decision trees, by contrast, are inherently designed around **axis-aligned recursive partitioning**, which is precisely the right inductive bias for tabular data where features are semantically distinct and independently meaningful. A split on `rsi_14 < 35` is a natural, interpretable financial rule; the equivalent operation in a deep network requires an entire sub-circuit to be synthesised from random initialisations.

Grinsztajn et al. (2022) formalised this intuition empirically: across 45 heterogeneous tabular benchmarks, GBDTs dominated MLPs in 67% of datasets, with the gap widening on financial and scientific datasets characterised by **irregular feature distributions and sparse predictive signals** — precisely the conditions that obtain in equity factor research.

#### 3.1.2 Data Efficiency and the Signal-to-Noise Ratio

Financial return data is famously low signal-to-noise. The annualised Sharpe ratio of most alpha factors is well below 1.0; the typical Information Coefficient (IC) of a single technical factor is between 0.02 and 0.06 — a signal that would be invisible to a human observer in a scatter plot. Deep learning models require large sample sizes to generalise reliably; their sample complexity scales with architectural depth and parameter count.

A universe of 500 stocks × 2,500 daily observations = 1.25M samples sounds large, but after applying forward-window targets, purge/embargo periods, and train/test splits, the effective training set available per fold may be closer to 250K–400K heterogeneous observations. GBDTs achieve competitive variance-bias trade-offs at this scale without regularisation complexity, while deep models routinely overfit, requiring extensive architectural search, dropout tuning, and batch-normalisation configuration that introduces researcher degrees of freedom.

#### 3.1.3 Robustness to Non-Stationarity

Financial time series are **I(1) or higher-order integrated processes** in price space, and exhibit volatility clustering, fat-tailed return distributions (excess kurtosis empirically ~5–7 for daily returns), and regime-switching behaviour. Feature distributions shift materially across market cycles: the Z-score of volatility in the 2020 COVID drawdown is statistically incomparable to its 2017 distribution.

Deep learning optimised with Adam or RMSProp is susceptible to **catastrophic forgetting and implicit distribution shift**, where new gradient updates overwrite representations learned from older regimes. GBDTs trained with subsampling, column subsampling, and regularised leaf weights are inherently more robust to distributional shift: each tree is a weak learner that contributes only a small, bounded update, and the ensemble's predictions decay gracefully under covariate shift.

#### 3.1.4 Hyperparameter Sensitivity and Operational Stability

A production quant fund requires a model that can be retrained monthly or weekly on rolling data without manual intervention. Deep network hyperparameter spaces — learning rate schedules, architecture (depth, width, skip connections), dropout rates, batch sizes, optimiser choice — require extensive search to reproduce stable performance. GBDTs have far fewer hyperparameters, and their performance is known to be **monotone in `n_estimators`** (with early stopping) and relatively insensitive to `max_depth` within a modest range. This operational stability is a first-order consideration in production deployment.

#### 3.1.5 Exact SHAP Tractability

Perhaps the most decisive practical argument is explainability. Regulatory frameworks (MiFID II, Dodd-Frank, Basel III internal model approval) and risk committees increasingly require that algorithmic trading strategies provide **ex-ante, factor-level explanations** for position sizing. For deep networks, SHAP values require sampling-based SHAP approximations (KernelSHAP or DeepSHAP), which are computationally expensive, statistically noisy, and architecturally dependent.

For tree ensembles, `shap.TreeExplainer` computes **exact Shapley values** in polynomial time using the path-dependent algorithm of Lundberg et al. (2020). This is not a minor convenience — it is a categorical difference that makes tree-based alpha models the only class of model for which rigorous, exact interpretability is operationally tractable on a live production cadence.

---

### 3.2 SHAP: Game-Theoretic Feature Attribution for Ensemble Models

#### 3.2.1 The Shapley Framework

SHAP is grounded in cooperative game theory, specifically the **Shapley value** of Shapley (1953). Given a model `f` with `M` features, the SHAP value `φᵢ(x)` for feature `i` at observation `x` is defined as:

```
φᵢ(x) = Σ_{S ⊆ F \ {i}} [|S|! (M - |S| - 1)! / M!] · [f(S ∪ {i}) - f(S)]
```

where `F` is the full feature set, `S` is a coalition of features (a subset not including `i`), and the sum is taken over all `2^(M-1)` such coalitions. The term `f(S)` denotes the model output when only the features in coalition `S` are observed (the remaining features are marginalised out under some baseline distribution).

This formulation satisfies four mathematically desirable axioms:
- **Efficiency**: `Σ φᵢ(x) = f(x) - E[f(X)]` — attributions sum exactly to the prediction deviation from baseline.
- **Symmetry**: Features that make identical contributions to all coalitions receive identical SHAP values.
- **Dummy**: A feature that never changes the model output receives a SHAP value of zero.
- **Linearity/Additivity**: SHAP values for a linear combination of models are the linear combination of individual SHAP values.

No other additive feature attribution method satisfies all four axioms simultaneously (Lundberg & Lee, 2017).

#### 3.2.2 TreeExplainer: Exact Computation via Tree Path Integration

For an ensemble of `T` trees each with maximum depth `D` and maximum `L` leaves, Lundberg et al. (2020) derive an exact algorithm that computes all `M` Shapley values for a single observation in `O(TLD²)` time, compared to the exponential `O(2^M)` naive computation. The algorithm tracks how the expected model output changes as features are progressively revealed along each tree path, accumulating attribution mass at each split node.

The `interventional` perturbation mode (used in this pipeline) marginalises missing features by conditioning on a representative background dataset rather than on tree structure, producing attributions that respect the true joint feature distribution and avoid spurious attributions from the tree's training-data-dependent split structure.

#### 3.2.3 SHAP in Quant Finance: Beyond Feature Importance

The SHAP framework enables three distinct use cases in the production alpha pipeline:

1. **Global Factor Attribution** (Summary Plots): The distribution of SHAP values across the full panel reveals which factors drive aggregate model behaviour and their directional effects. A negative SHAP for `zscore_20d` confirms the model has learned a mean-reversion signal; a positive SHAP for `mom_21d` confirms cross-sectional momentum.

2. **Conditional Factor Interaction** (Dependence Plots): SHAP dependence plots visualise the partial function `φᵢ(x) ~ g(xᵢ, xⱼ*)` where `xⱼ*` is the automatically selected interaction feature. This reveals non-linearities and regime-conditional effects that would be invisible in linear factor attribution.

3. **Trade-Level Explainability** (Waterfall Plots): For any individual position, the waterfall plot provides an ordered decomposition from the expected model output `E[f(X)]` to the specific prediction `f(x)`, with each feature's contribution colour-coded by direction. This is the auditable explanation that satisfies compliance requirements.

---

### 3.3 Purged Time-Series Cross-Validation: Rigorous OOS Estimation Under Temporal Dependence

#### 3.3.1 The Leakage Problem in Financial ML

Standard cross-validation — whether `k-fold`, stratified, or repeated — assumes that the observations in the dataset are **independently and identically distributed (i.i.d.)**. This assumption is violated, often catastrophically, in financial time-series data.

Consider a dataset of daily equity returns with a 21-day rolling momentum feature. If standard k-fold CV is applied, observation `t` may appear in the test set while observation `t-1` (which shares 20/21 days of the same feature window) appears in the training set. The model can effectively "memorise" tomorrow's return by exploiting the near-identical features of nearby observations — a form of temporal data leakage that produces OOS metrics that are optimistically biased by one to two orders of magnitude relative to live performance.

For a 5-day forward return target, the target at time `t` overlaps with the feature window at time `t+1` through `t+4`. Including these observations in the training set constitutes **direct label leakage** — the model sees the answer while learning.

#### 3.3.2 Architecture of PurgedTimeSeriesSplit

The `PurgedTimeSeriesSplit` class implemented in `src/ml_pipeline.py` extends `sklearn.model_selection.BaseCrossValidator` with three temporal integrity mechanisms:

**Mechanism 1: Strict Walk-Forward Ordering**

For each fold `k`, all training observations are strictly earlier in calendar time than all test observations:

```
[─── TRAIN FOLD k ───][GAP][─── TEST FOLD k ───][EMBARGO][─→ future ─→]
```

No observation in the test set can have its index precede any observation in the training set. This eliminates the primary source of look-ahead bias.

**Mechanism 2: Purge Gap**

A contiguous block of `purge_gap` observations is removed from the boundary between the training set and the test set. The purge gap must satisfy:

```
purge_gap ≥ max(lookback_window, forward_horizon) - 1
```

In this pipeline, `purge_gap = forward_days + 1` at minimum. This ensures that no training-set observation's feature window overlaps with any test-set observation's target window. Without this purge, the trailing training observations would contain return data from the test period embedded in their rolling features.

**Mechanism 3: Embargo Period**

An embargo of `embargo_pct` of the total sample is applied **after** the test set. This prevents a subtle but real leakage channel: serial autocorrelation in residuals means that the model could implicitly learn to exploit the autocorrelation structure of the test-period errors to improve performance on the next immediately subsequent test period. The embargo breaks this by creating a buffer that is excluded from all CV folds.

#### 3.3.3 The Walk-Forward Structure Visualised

```
Timeline →  2015          2017          2019          2021          2023    2024

Fold 1:     [TRAIN══════════════════][P][TEST═══]
Fold 2:     [TRAIN══════════════════════════][P][TEST═══]
Fold 3:     [TRAIN══════════════════════════════════][P][TEST═══]
Fold 4:     [TRAIN══════════════════════════════════════════][P][TEST═══]
Fold 5:     [TRAIN══════════════════════════════════════════════════][P][TEST]

[P] = Purge Gap (purge_gap periods)
Embargo applied after each TEST block (not shown for clarity)
```

Each successive fold expands the training set by one test block, ensuring that the model trained on Fold `k+1` has seen all data available by the end of Fold `k`'s test period. This mirrors the operational reality of a production model that is retrained on an expanding window of history.

#### 3.3.4 Information Coefficient as the Primary Evaluation Metric

We report the **Spearman rank Information Coefficient (IC)** as the primary OOS performance metric rather than RMSE or R². The IC is defined as:

```
IC = ρₛ(f(Xₜₑₛₜ), yₜₑₛₜ)
```

where `ρₛ` denotes the Spearman rank correlation. The IC is preferred because:

1. It is **rank-based**, eliminating the influence of outlier returns that dominate MSE/RMSE in heavy-tailed distributions.
2. It directly measures the model's ability to **rank securities cross-sectionally** — the operative quantity in a market-neutral long-short equity strategy, where only relative predictions matter.
3. An IC of 0.05 is often sufficient to generate economically significant alpha in a diversified portfolio context (Grinold & Kahn, 2000).

---

## 4. Feature Engineering Methodology

All features are computed by `src/feature_engineering.py` via the `FeatureEngineer` class. The full feature taxonomy is:

| Factor Group | Feature Name | Formula / Description |
|---|---|---|
| **Momentum** | `mom_{N}d` | Σ log-returns over N ∈ {5, 10, 21, 63} days |
| **Mean Reversion** | `zscore_{N}d` | (Pₜ − μₙ) / σₙ over N ∈ {20, 60} days |
| **Oscillator** | `rsi_14` | Wilder EMA-smoothed RSI with span=14 |
| **Trend** | `macd_line` | EMA(12) − EMA(26) |
| **Trend** | `macd_signal` | EMA(9) of MACD line |
| **Trend** | `macd_hist` | MACD line − Signal line |
| **Realised Vol** | `ann_vol_{N}d` | σ(log-ret, N) × √252 for N ∈ {5, 10, 21} |
| **Vol Regime** | `vol_ratio_5_21` | σ(5d) / σ(21d) — short-term vol vs. medium-term |
| **Microstructure** | `atr_14` | ATR(14) normalised by closing price |
| **Volume** | `volume_zscore_20d` | Z-score of log-volume over 20-day rolling window |
| **Volume** | `dollar_vol_zscore_20d` | Z-score of log(Close × Volume) over 20 days |

All features are computed in a single vectorised pass per security and then stacked into a panel (multi-security cross-section). `RobustScaler` (median and IQR normalisation) is applied within each CV fold to prevent scale-based leakage from the test set into the training set's normalisation parameters.

---

## 5. Installation & Usage

```bash
# Clone the repository
git clone https://github.com/your-org/alpha-factor-combiner.git
cd alpha-factor-combiner

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the research notebook (requires Jupyter)
pip install jupyter
jupyter notebook research_notebook.md
```

**Programmatic Usage:**

```python
import yfinance as yf
import pandas as pd
from src.feature_engineering import FeatureEngineer
from src.ml_pipeline import AlphaModel
from src.explainer import ModelExplainer

# 1. Download data
df = yf.download("AAPL", start="2018-01-01", end="2024-12-31", auto_adjust=True)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

# 2. Engineer features and target
fe = FeatureEngineer()
X = fe.fit_transform(df)
y = FeatureEngineer.build_target(df["Close"], forward_days=5)

# 3. Train with purged CV
model = AlphaModel(n_splits=5, purge_gap=6)
result = model.train(X, y)
print(f"Mean OOS IC: {result.mean_oos_ic:.4f}")

# 4. Explain with SHAP
X_scaled = pd.DataFrame(
    result.scaler.transform(X.dropna()),
    index=X.dropna().index, columns=X.dropna().columns
)
explainer = ModelExplainer(result.model, result.feature_names, output_dir="outputs/shap")
explainer.fit(X_scaled.sample(300, random_state=42))
explainer.explain(X_scaled)
explainer.plot_summary(X_scaled)
```

---

## 6. Configuration Reference

| Parameter | Class | Default | Description |
|---|---|---|---|
| `price_col` | `FeatureEngineer` | `"Close"` | Column name for closing prices |
| `volume_col` | `FeatureEngineer` | `"Volume"` | Column name for volume |
| `forward_days` | `build_target()` | `5` | Prediction horizon in trading days |
| `n_splits` | `AlphaModel` | `5` | Number of CV folds |
| `purge_gap` | `AlphaModel` | `5` | Periods to purge at train/test boundary |
| `embargo_pct` | `AlphaModel` | `0.01` | Fraction of sample used as embargo |
| `early_stopping_rounds` | `AlphaModel` | `50` | XGBoost early stopping patience |
| `n_estimators` | XGBoost | `500` | Maximum boosting rounds |
| `max_depth` | XGBoost | `4` | Maximum tree depth (keep ≤ 5 for finance) |
| `learning_rate` | XGBoost | `0.02` | Shrinkage / step size |
| `MAX_DISPLAY` | `ModelExplainer` | `20` | Max features in SHAP summary plots |

---

## 7. Empirical Considerations & Known Limitations

- **Universe Bias**: The default 5-stock tech universe exhibits pronounced co-movement (average pairwise correlation > 0.70). Production deployment should use a diversified cross-section of 100–500 securities across sectors to provide genuine cross-sectional signal.
- **Survivorship Bias**: `yfinance` downloads current-index constituents only. Point-in-time universe construction requires a commercial data vendor (e.g., Compustat, Bloomberg).
- **Transaction Costs**: All reported metrics are gross of trading costs. At 5-day rebalance frequency, round-trip costs of 5–10 bps should be deducted from the Q5–Q1 spread before economic conclusions are drawn.
- **Non-Stationarity**: Features are computed on raw price and return series. While log-returns are approximately stationary, price-level features (MACD, absolute ATR) remain non-stationary. Consider normalising by rolling volatility or applying percentage-rank transforms within cross-sections.
- **Parameter Sensitivity**: The purge gap must be at least `max(max_lookback_window, forward_horizon)`. Underspecifying the purge gap will produce optimistically biased OOS ICs that do not translate to live performance.

---

## 8. References

- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS 30.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2, 56–67.
- Prado, M. L. de. (2018). *Advances in Financial Machine Learning.* Wiley.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why tree-based models still outperform deep learning on tabular data.* NeurIPS 35.
- Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* (2nd ed.). McGraw-Hill.
- Shapley, L. S. (1953). *A value for n-person games.* In Kuhn, H.W. & Tucker, A.W. (Eds.), Contributions to the Theory of Games II. Princeton University Press.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD '16.

---

*This repository is intended for research and educational purposes. Nothing contained herein constitutes financial advice or a solicitation to trade securities.*
