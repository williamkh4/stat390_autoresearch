"""
Feature engineering for daily Victoria demand.

Each helper adds a clearly-named group of columns to the merged panel.
The AutoResearch loop toggles groups on/off via FeatureConfig so that feature
sets become part of the search space, not hard-coded into one script.

Lag/rolling features are computed on the *full* panel but any row whose lag
reaches into the future is guarded by shifting first and dropping NaNs at
fit time. Importantly, lag windows never cross the train/test boundary at
prediction time because the loop uses `.predict` on out-of-sample rows whose
lag values are historical (i.e., already observed training data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd


TARGET_COL = "demand"


@dataclass
class FeatureConfig:
    """Switch groups of features on/off. Used as a candidate knob by the loop."""
    use_calendar: bool = True
    use_temp: bool = True              # Kaggle min/max/solar/rain
    use_apparent_temp: bool = True     # Open-Meteo apparent_temp (project hypothesis)
    use_rrp: bool = True               # Actual RRP as a feature (upper bound: not predicted RRP)
    demand_lags: List[int] = field(default_factory=lambda: [1, 7])
    rolling_windows: List[int] = field(default_factory=lambda: [7])

    def describe(self) -> str:
        flags = [
            f"cal={int(self.use_calendar)}",
            f"temp={int(self.use_temp)}",
            f"apptemp={int(self.use_apparent_temp)}",
            f"rrp={int(self.use_rrp)}",
            f"lags={self.demand_lags}",
            f"roll={self.rolling_windows}",
        ]
        return " ".join(flags)


def _add_calendar(df: pd.DataFrame) -> List[str]:
    """Add calendar columns; return the names of columns added."""
    t = df["time"]
    df["dow"] = t.dt.dayofweek          # 0=Mon
    df["month"] = t.dt.month
    df["day_of_year"] = t.dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Smooth annual cycle (captures seasonality without a huge one-hot).
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return ["dow", "month", "day_of_year", "is_weekend", "doy_sin", "doy_cos",
            "school_day", "holiday"]


def _add_lags(df: pd.DataFrame, lags: List[int], rolls: List[int]) -> List[str]:
    cols: List[str] = []
    for lag in lags:
        name = f"demand_lag{lag}"
        df[name] = df[TARGET_COL].shift(lag)
        cols.append(name)
    for w in rolls:
        # Use shift(1) so the rolling mean at time t does NOT include y_t itself.
        name = f"demand_roll{w}"
        df[name] = df[TARGET_COL].shift(1).rolling(w, min_periods=w).mean()
        cols.append(name)
    return cols


def build_features(df: pd.DataFrame, config: FeatureConfig):
    """
    Return (X, y, feature_names).

    Rows containing any NaN in the selected feature set are dropped (this
    happens naturally at the start of the series from the lag warmup).
    """
    df = df.copy()
    feature_cols: List[str] = []

    if config.use_calendar:
        feature_cols += _add_calendar(df)

    if config.use_temp:
        feature_cols += [
            "min_temperature", "max_temperature", "solar_exposure", "rainfall",
            "temp_mean", "temp_max", "temp_min",
            "sunshine_s", "precip_mm", "daylight_s",
        ]

    if config.use_apparent_temp:
        feature_cols += ["apparent_temp_mean", "apparent_temp_max", "apparent_temp_min"]

    if config.use_rrp:
        feature_cols += ["RRP"]

    if config.demand_lags or config.rolling_windows:
        feature_cols += _add_lags(df, config.demand_lags, config.rolling_windows)

    # Keep feature order stable + drop duplicate columns in case of overlap.
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    keep = ["time", TARGET_COL] + feature_cols
    df = df[keep].dropna().reset_index(drop=True)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    times = df["time"].reset_index(drop=True)
    return X, y, feature_cols, times
