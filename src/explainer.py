from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


class ModelExplainer:
    PLOT_DPI: int = 150
    PLOT_FIGSIZE: Tuple[int, int] = (12, 8)
    MAX_DISPLAY: int = 20

    def __init__(self, model: XGBRegressor, feature_names: List[str],
                 output_dir: Union[str, Path] = "outputs/shap") -> None:
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._explainer: Optional[shap.TreeExplainer] = None
        self._shap_values: Optional[np.ndarray] = None
        self._X_background: Optional[pd.DataFrame] = None

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None: raise RuntimeError("Call .fit() first.")
        return self._explainer

    @property
    def shap_values(self) -> np.ndarray:
        if self._shap_values is None: raise RuntimeError("Call .explain() first.")
        return self._shap_values

    def fit(self, X_background: pd.DataFrame) -> ModelExplainer:
        self._X_background = X_background[self.feature_names].copy()
        self._explainer = shap.TreeExplainer(
            self.model, data=self._X_background, feature_perturbation="interventional", model_output="raw"
        )
        return self

    def explain(self, X: pd.DataFrame, check_additivity: bool = False) -> np.ndarray:
        self._shap_values = np.array(
            self.explainer.shap_values(X[self.feature_names], check_additivity=check_additivity))
        return self._shap_values

    def get_shap_dataframe(self, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(self.shap_values, index=index, columns=self.feature_names)

    def mean_absolute_shap(self) -> pd.Series:
        return pd.DataFrame(np.abs(self.shap_values), columns=self.feature_names).mean().sort_values(
            ascending=False).rename("mean_abs_shap")

    def _save_fig(self, filename: str) -> Path:
        out_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.PLOT_DPI, bbox_inches="tight")
        plt.close("all")
        return out_path

    def plot_summary(self, X: pd.DataFrame, plot_type: str = "dot", filename: str = "shap_summary.png") -> Path:
        fig, ax = plt.subplots(figsize=self.PLOT_FIGSIZE)
        shap.summary_plot(self.shap_values, X[self.feature_names], plot_type=plot_type, max_display=self.MAX_DISPLAY,
                          show=False)
        plt.title("SHAP Feature Importance — Cross-Sectional Alpha", fontsize=13, fontweight="bold", pad=12)
        return self._save_fig(filename)

    def plot_bar_summary(self, filename: str = "shap_bar_summary.png") -> Path:
        fig, ax = plt.subplots(figsize=self.PLOT_FIGSIZE)
        shap.summary_plot(self.shap_values, feature_names=self.feature_names, plot_type="bar",
                          max_display=self.MAX_DISPLAY, show=False)
        plt.title("Mean |SHAP| — Global Feature Importance", fontsize=13, fontweight="bold", pad=12)
        return self._save_fig(filename)

    def plot_dependence(self, feature: str, interaction_feature: str = "auto", X: Optional[pd.DataFrame] = None,
                        filename: Optional[str] = None) -> Path:
        fig, ax = plt.subplots(figsize=self.PLOT_FIGSIZE)
        bg = X[self.feature_names] if X is not None else pd.DataFrame(np.zeros((1, len(self.feature_names))),
                                                                      columns=self.feature_names)
        shap.dependence_plot(feature, self.shap_values, bg, interaction_index=interaction_feature, ax=ax, show=False)
        ax.set_title(f"SHAP Dependence Plot — {feature}", fontsize=13, fontweight="bold", pad=12)
        return self._save_fig(filename or f"shap_dependence_{feature.replace('/', '_')}.png")

    def plot_waterfall(self, sample_idx: int = 0, filename: Optional[str] = None) -> Path:
        expected_value = float(self.explainer.expected_value[0]) if isinstance(self.explainer.expected_value, (list,
                                                                                                               np.ndarray)) else self.explainer.expected_value
        shap_exp = shap.Explanation(values=self.shap_values[sample_idx], base_values=expected_value,
                                    feature_names=self.feature_names)
        fig, ax = plt.subplots(figsize=self.PLOT_FIGSIZE)
        shap.plots.waterfall(shap_exp, show=False, max_display=15)
        plt.title(f"SHAP Waterfall — Sample #{sample_idx}", fontsize=13, fontweight="bold", pad=12)
        return self._save_fig(filename or f"shap_waterfall_sample_{sample_idx}.png")

    def plot_feature_importance_comparison(self, xgb_gain_importance: pd.Series,
                                           filename: str = "importance_comparison.png") -> Path:
        shap_imp = self.mean_absolute_shap()
        common = shap_imp.index.intersection(xgb_gain_importance.index)
        shap_norm, gain_norm = shap_imp[common] / shap_imp[common].sum(), xgb_gain_importance[common] / \
                               xgb_gain_importance[common].sum()
        top_n = min(15, len(common))
        top_features = shap_norm.nlargest(top_n).index

        fig, ax = plt.subplots(figsize=(14, 7))
        x, width = np.arange(top_n), 0.38
        ax.bar(x - width / 2, shap_norm[top_features], width, label="SHAP (Mean |φ|)", color="#2196F3", alpha=0.85)
        ax.bar(x + width / 2, gain_norm[top_features], width, label="XGBoost Gain", color="#FF5722", alpha=0.85)

        ax.set_xticks(x);
        ax.set_xticklabels(top_features, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Normalised Importance", fontsize=11)
        ax.set_title("Feature Importance: SHAP vs. XGBoost Gain (Normalised)", fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=10);
        ax.spines["top"].set_visible(False);
        ax.spines["right"].set_visible(False)
        return self._save_fig(filename)
