"""
Reference baseline for the project: Seasonal Naive with weekly period.

y_hat(t) = demand(t - 7)

Rationale: daily electricity demand in Victoria has a strong weekly cycle
(weekdays vs weekends) and a weaker intra-week pattern. A 7-day lag is the
standard seasonal-naive choice and is well-known to be hard to beat without
real feature work. The loop's goal is to drive validation MSE below this
number.

The class uses sklearn-style fit/predict purely to stay interchangeable with
the other candidates. For seasonal naive there's nothing to fit; we just
stash the last observed `period` values from the training history so that
predictions at the start of the validation window remain valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    """Seasonal naive forecaster: predict demand[t] = demand[t - period]."""

    def __init__(self, period: int = 7):
        self.period = period
        self.history_: pd.Series | None = None

    # The baseline is time-aware, so it takes (time, y) rather than (X, y).
    def fit(self, times: pd.Series, y: np.ndarray) -> "SeasonalNaive":
        self.history_ = pd.Series(np.asarray(y, dtype=float),
                                  index=pd.to_datetime(times))
        return self

    def predict(self, times: pd.Series) -> np.ndarray:
        if self.history_ is None:
            raise RuntimeError("Baseline not fit yet")
        times = pd.to_datetime(times)
        preds = np.empty(len(times))
        for i, t in enumerate(times):
            target = t - pd.Timedelta(days=self.period)
            if target in self.history_.index:
                preds[i] = self.history_.loc[target]
            else:
                # Fallback: use the overall training mean. Only happens if
                # the validation window starts fewer than `period` days after
                # the training window ends, which we avoid by construction.
                preds[i] = float(self.history_.mean())
        return preds

    def __repr__(self) -> str:
        return f"SeasonalNaive(period={self.period})"
