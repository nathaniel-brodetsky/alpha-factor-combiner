from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


class FeatureEngineer:
    RSI_WINDOW: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    VOL_WINDOWS: List[int] = [5, 10, 21]
    ZSCORE_WINDOWS: List[int] = [20, 60]
    MOMENTUM_WINDOWS: List[int] = [5, 10, 21, 63]

    def __init__(self, price_col: str = "Close", volume_col: str = "Volume") -> None:
        self.price_col = price_col
        self.volume_col = volume_col
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._build_features(df)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df[self.price_col].astype(np.float64)
        volume = df[self.volume_col].astype(np.float64) if self.volume_col in df.columns else None
        log_ret = np.log(close).diff()

        frames: List[pd.Series] = []
        frames.extend(self._momentum_features(log_ret))
        frames.extend(self._rolling_zscore_features(close))
        frames.append(self._rsi(close))
        frames.extend(self._macd_features(close))
        frames.extend(self._volatility_features(log_ret))
        frames.extend(self._volume_features(close, volume))
        frames.append(self._atr(df))

        feature_df = pd.concat(frames, axis=1)
        self._feature_names = feature_df.columns.tolist()
        return feature_df

    def _momentum_features(self, log_ret: pd.Series) -> List[pd.Series]:
        return [log_ret.rolling(w).sum().rename(f"mom_{w}d") for w in self.MOMENTUM_WINDOWS]

    def _rolling_zscore_features(self, close: pd.Series) -> List[pd.Series]:
        features = []
        for w in self.ZSCORE_WINDOWS:
            roll = close.rolling(w)
            features.append(((close - roll.mean()) / roll.std()).rename(f"zscore_{w}d"))
        return features

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain, loss = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(com=self.RSI_WINDOW - 1, min_periods=self.RSI_WINDOW).mean()
        avg_loss = loss.ewm(com=self.RSI_WINDOW - 1, min_periods=self.RSI_WINDOW).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).rename("rsi_14")

    def _macd_features(self, close: pd.Series) -> List[pd.Series]:
        ema_fast = close.ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=self.MACD_SLOW, adjust=False).mean()
        macd_line = (ema_fast - ema_slow).rename("macd_line")
        signal_line = macd_line.ewm(span=self.MACD_SIGNAL, adjust=False).mean().rename("macd_signal")
        return [macd_line, signal_line, (macd_line - signal_line).rename("macd_hist")]

    def _volatility_features(self, log_ret: pd.Series) -> List[pd.Series]:
        features = [(log_ret.rolling(w).std() * np.sqrt(252)).rename(f"ann_vol_{w}d") for w in self.VOL_WINDOWS]
        vol_ratio = (log_ret.rolling(self.VOL_WINDOWS[0]).std() /
                     log_ret.rolling(self.VOL_WINDOWS[2]).std().replace(0.0, np.nan)).rename("vol_ratio_5_21")
        features.append(vol_ratio)
        return features

    def _volume_features(self, close: pd.Series, volume: Optional[pd.Series]) -> List[pd.Series]:
        if volume is None: return []
        log_vol = np.log(volume.replace(0.0, np.nan))
        vol_z = ((log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()).rename("volume_zscore_20d")
        dollar_vol = close * volume
        dv_z = ((dollar_vol - dollar_vol.rolling(20).mean()) / dollar_vol.rolling(20).std()).rename(
            "dollar_vol_zscore_20d")
        return [vol_z, dv_z]

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        if not {"High", "Low", "Close"}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index, name="atr_14")
        high, low, close = df["High"].astype(np.float64), df["Low"].astype(np.float64), df["Close"].astype(np.float64)
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(com=self.RSI_WINDOW - 1, min_periods=self.RSI_WINDOW).mean()
        return (atr / close.replace(0.0, np.nan)).rename("atr_14")

    @staticmethod
    def build_target(close: pd.Series, forward_days: int = 5, log_return: bool = True) -> pd.Series:
        if log_return:
            return np.log(close.shift(-forward_days) / close).rename(f"fwd_log_ret_{forward_days}d")
        return (close.shift(-forward_days) / close - 1.0).rename(f"fwd_ret_{forward_days}d")
