from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


@dataclass
class CVFold:
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_period: Tuple[Any, Any]
    test_period: Tuple[Any, Any]
    oos_rmse: float = np.nan
    oos_ic: float = np.nan


@dataclass
class TrainingResult:
    model: XGBRegressor
    scaler: RobustScaler
    feature_names: List[str]
    cv_folds: List[CVFold]
    mean_oos_rmse: float
    mean_oos_ic: float
    best_iteration: int
    params: Dict[str, Any]


class PurgedTimeSeriesSplit(BaseCrossValidator):

    def __init__(self, n_splits: int = 5, purge_gap: int = 5, embargo_pct: float = 0.01,
                 test_size: Optional[int] = None) -> None:
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.test_size = test_size

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[np.ndarray] = None) -> Generator[
        Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        embargo_size = int(np.floor(n_samples * self.embargo_pct))
        test_size = self.test_size or int(np.floor(n_samples / (self.n_splits + 1)))
        indices = np.arange(n_samples)
        fold_starts = np.linspace(test_size, n_samples - test_size, self.n_splits, dtype=int)

        for fold_start in fold_starts:
            test_start, test_end = fold_start, min(fold_start + test_size, n_samples)
            train_end = test_start - self.purge_gap - embargo_size
            if train_end <= 0: continue
            train_idx, test_idx = indices[:train_end], indices[test_start:test_end]
            if len(train_idx) == 0 or len(test_idx) == 0: continue
            yield train_idx, test_idx

    def get_n_splits(self, X: Optional[Any] = None, y: Optional[Any] = None, groups: Optional[Any] = None) -> int:
        return self.n_splits


class AlphaModel:
    DEFAULT_XGB_PARAMS = {
        "n_estimators": 500, "max_depth": 4, "learning_rate": 0.02,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 10,
        "gamma": 0.1, "reg_alpha": 0.05, "reg_lambda": 1.0,
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "tree_method": "hist", "random_state": 42, "n_jobs": -1,
    }

    def __init__(self, xgb_params: Optional[Dict[str, Any]] = None, n_splits: int = 5, purge_gap: int = 5,
                 embargo_pct: float = 0.01, early_stopping_rounds: int = 50) -> None:
        self.xgb_params = {**self.DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.n_splits, self.purge_gap, self.embargo_pct = n_splits, purge_gap, embargo_pct
        self.early_stopping_rounds = early_stopping_rounds
        self._result: Optional[TrainingResult] = None

    @property
    def result(self) -> TrainingResult:
        if self._result is None: raise RuntimeError("Model not trained. Call .train() first.")
        return self._result

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        X_clean, y_clean = self._align_and_drop_na(X, y)
        self._validate_temporal_index(X_clean)

        cv = PurgedTimeSeriesSplit(n_splits=self.n_splits, purge_gap=self.purge_gap, embargo_pct=self.embargo_pct)
        cv_folds: List[CVFold] = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_clean)):
            cv_folds.append(self._evaluate_fold(X_clean, y_clean, train_idx, test_idx, fold_idx))

        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), index=X_clean.index, columns=X_clean.columns)
        final_model = self._fit_model(X_scaled, y_clean, eval_set=None)

        self._result = TrainingResult(
            model=final_model, scaler=scaler, feature_names=X_clean.columns.tolist(),
            cv_folds=cv_folds, mean_oos_rmse=float(np.nanmean([f.oos_rmse for f in cv_folds])),
            mean_oos_ic=float(np.nanmean([f.oos_ic for f in cv_folds])),
            best_iteration=getattr(final_model, "best_iteration", self.xgb_params["n_estimators"]),
            params=self.xgb_params
        )
        return self._result

    def predict(self, X: pd.DataFrame) -> pd.Series:
        res = self.result
        X_scaled = pd.DataFrame(res.scaler.transform(X[res.feature_names]), index=X.index, columns=res.feature_names)
        return pd.Series(res.model.predict(X_scaled), index=X.index, name="alpha_score")

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        scores = self.result.model.get_booster().get_score(importance_type=importance_type)
        return pd.Series(scores, name=f"importance_{importance_type}").sort_values(ascending=False)

    def _evaluate_fold(self, X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray,
                       fold_idx: int) -> CVFold:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        scaler = RobustScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

        model = self._fit_model(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)])
        preds = model.predict(X_te_s)

        return CVFold(
            fold_idx=fold_idx, train_idx=train_idx, test_idx=test_idx,
            train_period=(X.index[train_idx[0]], X.index[train_idx[-1]]),
            test_period=(X.index[test_idx[0]], X.index[test_idx[-1]]),
            oos_rmse=float(np.sqrt(mean_squared_error(y_te, preds))),
            oos_ic=float(pd.Series(preds).corr(y_te.reset_index(drop=True), method="spearman"))
        )

    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]]) -> XGBRegressor:
        model = XGBRegressor(**self.xgb_params)
        fit_kwargs = {"verbose": False}
        if eval_set: fit_kwargs.update({"eval_set": eval_set, "early_stopping_rounds": self.early_stopping_rounds})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, **fit_kwargs)
        return model

    @staticmethod
    def _align_and_drop_na(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        combined = pd.concat([X, y], axis=1).dropna()
        return combined.iloc[:, :-1], combined.iloc[:, -1]

    @staticmethod
    def _validate_temporal_index(X: pd.DataFrame) -> None:
        if not isinstance(X.index, pd.DatetimeIndex) or not X.index.is_monotonic_increasing:
            raise ValueError("X must have a sorted DatetimeIndex for temporal CV.")
