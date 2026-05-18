"""
Stage-1 RRP predictor for the iter-2 two-stage pipeline.

H2 (predicted-RRP-as-feature) is the primary iter-2 hypothesis. Iter-1
could only test it via the leaky `full` feature preset that exposed
*observed* RRP at forecast time. This module replaces that by fitting a
small stage-1 model on weather + calendar + RRP lags and emitting a
*predicted* RRP column (`rrp_predicted`) that the stage-2 demand model
can consume cleanly.

Strict chronological fitting
============================
Stage-1 only sees rows that the caller marks as "train" via the
`train_mask` argument to `materialize_predicted_rrp`. For walk-forward
CV, the caller passes the fold's train mask; for the final
hold-out evaluation, the caller passes train+val as the fit window.
Either way, the predicted RRP for a row at time `t` is a function only
of features known at time `t` (no peeking at future RRP).

Feature set for stage-1 (deliberately narrow, no demand features):

  weather:    min_temperature, max_temperature, solar_exposure, rainfall,
              temp_mean, temp_max, temp_min,
              apparent_temp_mean, apparent_temp_max, apparent_temp_min
  calendar:   dow, month, day_of_year, is_weekend, doy_sin, doy_cos,
              school_day, holiday
  RRP lags:   RRP.shift(1), RRP.shift(7)

Stage-1 model default: GradientBoostingRegressor(n_estimators=300,
max_depth=3, learning_rate=0.05, random_state=0). Wrap in a
StandardScaler-equivalent? Not needed for GBM. Pass any sklearn-style
regressor via the constructor if you want to substitute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List
import numpy as np
import pandas as pd


WEATHER_COLS: List[str] = [
    "min_temperature", "max_temperature", "solar_exposure", "rainfall",
    "temp_mean", "temp_max", "temp_min",
    "apparent_temp_mean", "apparent_temp_max", "apparent_temp_min",
]
CALENDAR_COLS_RAW: List[str] = [
    "dow", "month", "day_of_year", "is_weekend", "doy_sin", "doy_cos",
    "school_day", "holiday",
]


def _ensure_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotently add the calendar columns stage-1 needs."""
    df = df.copy()
    if "dow" not in df.columns:
        df["dow"] = df["time"].dt.dayofweek
    if "month" not in df.columns:
        df["month"] = df["time"].dt.month
    if "day_of_year" not in df.columns:
        df["day_of_year"] = df["time"].dt.dayofyear
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
    if "doy_sin" not in df.columns:
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    if "doy_cos" not in df.columns:
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def _stage1_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the stage-1 feature frame, including RRP lags 1 and 7.

    Returns a frame with the same row order as `df` and a
    `_valid_mask` column = True iff every feature is non-NaN. Stage-1
    is fit on rows that are valid AND in `train_mask`; stage-1
    predicts everywhere `_valid_mask` is True (we forward-fill the
    head of the panel where RRP lags are still NaN).
    """
    df = _ensure_calendar_columns(df)
    out = df[["time", "RRP"]].copy()
    for c in WEATHER_COLS:
        out[c] = df[c]
    for c in CALENDAR_COLS_RAW:
        out[c] = df[c]
    out["RRP_lag1"] = df["RRP"].shift(1)
    out["RRP_lag7"] = df["RRP"].shift(7)

    feature_cols = WEATHER_COLS + CALENDAR_COLS_RAW + ["RRP_lag1", "RRP_lag7"]
    out["_valid_mask"] = out[feature_cols].notna().all(axis=1)
    out.attrs["feature_cols"] = feature_cols
    return out


def _default_factory() -> Any:
    """Default stage-1 model: a small GBM. sklearn is required for this."""
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0,
    )


class RRPPredictor:
    """Stage-1 model: predicts daily RRP from weather + calendar + RRP lags.

    Strict chronological fitting: RRP at day t is predicted from features
    known at day t (no demand features, no future RRP). Used by
    FeatureConfig.use_predicted_rrp via materialize_predicted_rrp().
    """

    def __init__(self, model_factory: Callable[[], Any] | None = None):
        self.model_factory = model_factory or _default_factory
        self._model: Any = None
        self._feature_cols: List[str] | None = None

    def fit(self, df_train: pd.DataFrame) -> "RRPPredictor":
        feats = _stage1_feature_frame(df_train)
        valid = feats[feats["_valid_mask"]].copy()
        if len(valid) < 30:
            raise ValueError(
                f"Stage-1 RRP fit window too small: {len(valid)} rows. "
                "Check that the train mask covers >30 days after lag warmup."
            )
        feature_cols = feats.attrs["feature_cols"]
        X = valid[feature_cols].to_numpy(dtype=float)
        y = valid["RRP"].to_numpy(dtype=float)
        self._model = self.model_factory()
        self._model.fit(X, y)
        self._feature_cols = feature_cols
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict RRP for every row of `df` whose features are non-NaN.

        Rows where features are NaN (the head of the panel, before
        RRP_lag7 exists) receive NaN predictions. Callers must drop
        those rows or forward-fill from the training period before
        consuming the prediction column.
        """
        if self._model is None or self._feature_cols is None:
            raise RuntimeError("RRPPredictor not fit yet")
        feats = _stage1_feature_frame(df)
        out = pd.Series(np.nan, index=df.index, name="rrp_predicted")
        if feats["_valid_mask"].any():
            X = feats.loc[feats["_valid_mask"], self._feature_cols].to_numpy(dtype=float)
            preds = self._model.predict(X)
            out.loc[feats["_valid_mask"].values] = preds
        return out


@dataclass
class Stage1Diagnostics:
    """Optional reporting handle for stage-1 quality on the fit window."""
    fit_window_rows: int
    fit_mse: float
    fit_baseline_mse: float       # "predict mean RRP on the fit window"

    def beats_trivial(self) -> bool:
        return self.fit_mse < self.fit_baseline_mse


def materialize_predicted_rrp(
    combined_df: pd.DataFrame,
    train_mask,
    model_factory: Callable[[], Any] | None = None,
    return_diagnostics: bool = False,
):
    """Fit `RRPPredictor` on train rows; predict RRP for every row.

    Args:
        combined_df: panel that includes train + val (and optionally test)
                     rows, with `time` and `RRP` columns + weather columns.
        train_mask:  boolean mask aligned with combined_df.index, True for
                     rows the stage-1 model is allowed to see during fit.
        model_factory: optional zero-arg callable returning a fresh
                     sklearn-style regressor. Defaults to GBM-300/3/0.05.
        return_diagnostics: if True, returns (series, Stage1Diagnostics)
                     for sanity-checking stage-1 vs the trivial baseline.

    Returns:
        pd.Series aligned with combined_df.index, named 'rrp_predicted'.
        Rows where the stage-1 features are NaN (panel head) keep NaN
        in the series; the head is naturally dropped later by
        build_features.dropna().

        If return_diagnostics=True, also returns Stage1Diagnostics for
        the fit-window MSE vs predict-the-mean RRP MSE.
    """
    if "RRP" not in combined_df.columns:
        raise ValueError("combined_df must contain an 'RRP' column for stage-1.")
    if not hasattr(train_mask, "__len__") or len(train_mask) != len(combined_df):
        raise ValueError(
            f"train_mask length {len(train_mask) if hasattr(train_mask, '__len__') else '?'} "
            f"!= combined_df length {len(combined_df)}"
        )
    train_df = combined_df.loc[train_mask].copy()
    predictor = RRPPredictor(model_factory=model_factory)
    predictor.fit(train_df)
    series = predictor.predict(combined_df)

    if return_diagnostics:
        # MSE of stage-1 on its own fit window (in-sample sanity, not val).
        fit_feats = _stage1_feature_frame(train_df)
        valid = fit_feats["_valid_mask"]
        X = fit_feats.loc[valid, predictor._feature_cols].to_numpy(dtype=float)
        y = train_df.loc[fit_feats.index[valid], "RRP"].to_numpy(dtype=float)
        y_hat = predictor._model.predict(X)
        fit_mse = float(np.mean((y - y_hat) ** 2))
        trivial = float(np.mean((y - y.mean()) ** 2))
        diag = Stage1Diagnostics(
            fit_window_rows=int(valid.sum()),
            fit_mse=fit_mse,
            fit_baseline_mse=trivial,
        )
        return series, diag
    return series


if __name__ == "__main__":
    # Sanity smoke-test: fit stage-1 on iter-1's train window, report MSE.
    from src.data_loader import load_merged
    from src.split import make_walk_forward_folds

    df = load_merged()
    folds = make_walk_forward_folds(df)
    fold = folds[len(folds) // 2]
    combined = pd.concat([fold.train, fold.val]).reset_index(drop=True)
    train_mask = combined["time"] <= fold.train["time"].max()

    series, diag = materialize_predicted_rrp(combined, train_mask,
                                             return_diagnostics=True)
    print(f"Stage-1 RRP predictor (fold {fold.fold_idx}):")
    print(f"  fit window rows: {diag.fit_window_rows}")
    print(f"  in-sample fit MSE     : {diag.fit_mse:,.2f}")
    print(f"  trivial-baseline MSE  : {diag.fit_baseline_mse:,.2f}")
    print(f"  beats trivial: {diag.beats_trivial()}")
    print(f"  predicted-RRP series : {series.notna().sum()} non-NaN of {len(series)}")
