"""
AutoResearch loop.

One "run" of the loop = fit and score every registered candidate on the
validation split, write a leaderboard, and return the best candidate's config.
The locked 365-day test set is never touched here.

A candidate is a (feature_config, model_factory) pair. Swapping in new
candidates is how you "iterate" the research loop between runs.

Budget accounting:
  * runtime_sec per candidate and total wall-clock for the run
  * fit_count (number of model fits) as a simple "budget unit"
  * peak memory and gpu are out of scope for this first version
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Any
import json
import platform
import time
import uuid

import numpy as np
import pandas as pd

from .baselines import SeasonalNaive
from .features import FeatureConfig, build_features
from .metrics import PRIMARY_METRIC_NAME, score_all
from .split import Splits


# ----------------------------------------------------------------------------
# Candidate definitions
# ----------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    feature_config: FeatureConfig
    model_factory: Callable[[], Any]   # returns a fresh sklearn-like estimator
    is_baseline: bool = False

    def describe(self) -> str:
        return f"{self.name} | features[{self.feature_config.describe()}]"


def baseline_candidates() -> List[Candidate]:
    """Baseline reference models. Depend only on numpy/pandas, not sklearn.

    These are the "numbers to beat". run_baseline.py uses this list directly
    so the baseline analysis is usable even without sklearn installed.
    """
    empty_fc = FeatureConfig(
        use_calendar=False, use_temp=False, use_apparent_temp=False,
        use_rrp=False, demand_lags=[], rolling_windows=[],
    )
    return [
        # -- Baseline 1: weekly seasonal naive, period = 7 days.
        # Uses last week's observation; "floor" baseline every model must beat.
        Candidate(
            name="seasonal_naive_7",
            feature_config=empty_fc,
            model_factory=lambda: SeasonalNaive(period=7),
            is_baseline=True,
        ),
        # -- Baseline 2: yearly-aligned seasonal naive, period = 52*7 = 364.
        # Same day-of-week, 52 weeks ago; captures the annual cycle trivially.
        # A real model must beat this to justify using weather + RRP features.
        Candidate(
            name="seasonal_naive_364",
            feature_config=empty_fc,
            model_factory=lambda: SeasonalNaive(period=364),
            is_baseline=True,
        ),
    ]


def default_candidates() -> List[Candidate]:
    """Seed set for the first AutoResearch iteration = baselines + real models.

    Keep this short and cheap: the whole point of measuring runtime/budget
    is that you can grow this list deliberately, not by accident.
    """
    # Lazy imports: only required when the AutoResearch loop runs (not when
    # run_baseline.py runs), so baseline analysis works without sklearn.
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    return baseline_candidates() + [
        # -- Ridge: calendar + temperature only (classic feature set)
        Candidate(
            name="ridge_cal_temp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=False,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: Ridge(alpha=1.0, random_state=0),
        ),
        # -- Ridge: + apparent temperature (tests the core project hypothesis)
        Candidate(
            name="ridge_cal_temp_apptemp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=True,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: Ridge(alpha=1.0, random_state=0),
        ),
        # -- Random Forest: same features as above
        Candidate(
            name="rf_cal_temp_apptemp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=True,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: RandomForestRegressor(
                n_estimators=200, max_depth=None, random_state=0, n_jobs=-1,
            ),
        ),
        # -- GBM: full feature stack including RRP (observed, not predicted)
        Candidate(
            name="gbm_full",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=True,
                use_rrp=True, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0,
            ),
        ),
    ]


# ----------------------------------------------------------------------------
# Scoring one candidate
# ----------------------------------------------------------------------------

@dataclass
class CandidateResult:
    name: str
    feature_config: Dict[str, Any]
    is_baseline: bool
    metrics: Dict[str, float]
    runtime_sec: float
    n_train: int
    n_val: int
    n_features: int
    error: str | None = None


def _is_baseline_model(model: Any) -> bool:
    return isinstance(model, SeasonalNaive)


def score_candidate(candidate: Candidate, splits: Splits) -> CandidateResult:
    t0 = time.perf_counter()
    try:
        model = candidate.model_factory()

        if _is_baseline_model(model):
            # Seasonal naive uses time + y directly.
            train_hist = pd.concat([splits.train, splits.val.iloc[:0]])  # train only
            model.fit(train_hist["time"], train_hist["demand"].to_numpy())
            y_pred = model.predict(splits.val["time"])
            y_true = splits.val["demand"].to_numpy(dtype=float)
            n_features = 0
            n_train = len(train_hist)
            n_val = len(splits.val)
        else:
            fc = candidate.feature_config

            # Fit feature frame on train+val concatenated, but only train rows
            # get used for fitting; val rows get used only for prediction.
            # This keeps lag/rolling windows contiguous across the boundary.
            combined = pd.concat([splits.train, splits.val]).reset_index(drop=True)
            boundary = splits.train["time"].max()

            X, y, feat_names, times = build_features(combined, fc)
            train_mask = times <= boundary
            val_mask = ~train_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_true = y_val
            n_features = len(feat_names)
            n_train = int(train_mask.sum())
            n_val = int(val_mask.sum())

        metrics = score_all(y_true, y_pred)
        runtime = time.perf_counter() - t0
        return CandidateResult(
            name=candidate.name,
            feature_config=asdict(candidate.feature_config),
            is_baseline=candidate.is_baseline,
            metrics=metrics,
            runtime_sec=runtime,
            n_train=n_train,
            n_val=n_val,
            n_features=n_features,
        )
    except Exception as exc:
        runtime = time.perf_counter() - t0
        return CandidateResult(
            name=candidate.name,
            feature_config=asdict(candidate.feature_config),
            is_baseline=candidate.is_baseline,
            metrics={},
            runtime_sec=runtime,
            n_train=0, n_val=0, n_features=0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ----------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------

@dataclass
class RunReport:
    run_id: str
    timestamp: str
    total_runtime_sec: float
    fit_count: int              # the "budget" spent this run
    n_candidates: int
    splits_summary: str
    results: List[CandidateResult] = field(default_factory=list)
    best_name: str | None = None
    best_metric: float | None = None
    baseline_metric: float | None = None
    improvement_vs_baseline: float | None = None
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_runtime_sec": self.total_runtime_sec,
            "fit_count": self.fit_count,
            "n_candidates": self.n_candidates,
            "splits_summary": self.splits_summary,
            "best_name": self.best_name,
            "best_metric": self.best_metric,
            "baseline_metric": self.baseline_metric,
            "improvement_vs_baseline": self.improvement_vs_baseline,
            "env": self.env,
            "results": [asdict(r) for r in self.results],
        }


def run_loop(
    splits: Splits,
    candidates: List[Candidate] | None = None,
    results_dir: Path | str = "experiments/results",
) -> RunReport:
    """Fit + score every candidate once; write leaderboard; return report."""
    candidates = candidates or default_candidates()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results: List[CandidateResult] = []
    for cand in candidates:
        print(f"[autoresearch] scoring: {cand.describe()}")
        res = score_candidate(cand, splits)
        if res.error:
            print(f"  ! error: {res.error}  ({res.runtime_sec:.2f}s)")
        else:
            print(f"  {PRIMARY_METRIC_NAME}={res.metrics[PRIMARY_METRIC_NAME]:.2f} "
                  f"({res.runtime_sec:.2f}s, n_features={res.n_features})")
        results.append(res)
    total_runtime = time.perf_counter() - t0

    # Build leaderboard: best (lowest) primary metric first, ignoring errors.
    ranked = sorted(
        [r for r in results if not r.error],
        key=lambda r: r.metrics[PRIMARY_METRIC_NAME],
    )
    best = ranked[0] if ranked else None
    baseline_results = [r for r in results if r.is_baseline and not r.error]
    # When multiple baselines are registered, compare against the strongest
    # one (lowest MSE) -- beating the weakest baseline is table stakes.
    best_baseline = (
        min(baseline_results, key=lambda r: r.metrics[PRIMARY_METRIC_NAME])
        if baseline_results else None
    )
    baseline_metric = best_baseline.metrics[PRIMARY_METRIC_NAME] if best_baseline else None
    improvement = None
    if best and baseline_metric is not None:
        improvement = baseline_metric - best.metrics[PRIMARY_METRIC_NAME]

    report = RunReport(
        run_id=str(uuid.uuid4())[:8],
        timestamp=pd.Timestamp.utcnow().isoformat(),
        total_runtime_sec=total_runtime,
        fit_count=len(results),
        n_candidates=len(candidates),
        splits_summary=splits.describe(),
        results=results,
        best_name=best.name if best else None,
        best_metric=best.metrics[PRIMARY_METRIC_NAME] if best else None,
        baseline_metric=baseline_metric,
        improvement_vs_baseline=improvement,
        env={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )

    # Write the run report as JSON + a compact leaderboard CSV.
    out_json = results_dir / f"run_{report.run_id}.json"
    with open(out_json, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    leaderboard = pd.DataFrame([
        {
            "name": r.name,
            "is_baseline": r.is_baseline,
            PRIMARY_METRIC_NAME: r.metrics.get(PRIMARY_METRIC_NAME, np.nan),
            "rmse_demand": r.metrics.get("rmse_demand", np.nan),
            "mae_demand": r.metrics.get("mae_demand", np.nan),
            "runtime_sec": r.runtime_sec,
            "n_features": r.n_features,
            "error": r.error or "",
        }
        for r in results
    ]).sort_values(PRIMARY_METRIC_NAME, na_position="last")
    leaderboard.to_csv(results_dir / f"run_{report.run_id}_leaderboard.csv", index=False)

    print()
    print(f"[autoresearch] run_id={report.run_id} total={total_runtime:.2f}s "
          f"budget={report.fit_count} fits")
    if best:
        print(f"[autoresearch] best: {report.best_name}  "
              f"{PRIMARY_METRIC_NAME}={report.best_metric:.2f}")
    if improvement is not None:
        sign = "BELOW" if improvement > 0 else "ABOVE"
        print(f"[autoresearch] best baseline: {best_baseline.name} "
              f"({baseline_metric:.2f}) -> best is {abs(improvement):.2f} "
              f"{sign} best baseline")
    print(f"[autoresearch] wrote {out_json.name} + leaderboard.csv to {results_dir}")

    return report
