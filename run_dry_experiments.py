"""
Five sequential dry-run experiments.

Each experiment:
  - includes both seasonal-naive baselines (consistent floor)
  - adds ONE new candidate that differs from the previous experiment's
    new candidate by exactly one knob (clean ablation)
  - calls run_loop, which writes its own run_<id>.json, appends to
    master_log.csv, and updates champion.json on improvement

Story across the 5 runs:
  E1: baselines only                       -> initial champion
  E2: + ols_calendar                       -> can a calendar-only OLS beat seasonal_naive_364?
  E3: + ols_calendar_temp                  -> does adding standard temperature help?
  E4: + ols_calendar_temp_apptemp          -> does apparent temperature add value?  (H1)
  E5: + ols_full (calendar + temp + apparent + lags + RRP)  -> full feature stack

Models used here are numpy-only so this script runs without sklearn. A
real iteration on a fully-equipped machine should swap NumpyOLS for sklearn's
Ridge / RandomForestRegressor / GradientBoostingRegressor.

Usage:
    python run_dry_experiments.py
"""

from __future__ import annotations

from pathlib import Path

from src.autoresearch import (
    Candidate,
    baseline_candidates,
    run_loop,
)
from src.data_loader import load_merged, validate_merged
from src.features import FeatureConfig
from src.numpy_models import NumpyOLS
from src.split import make_splits


def _ols_calendar() -> Candidate:
    return Candidate(
        name="ols_calendar",
        feature_config=FeatureConfig(
            use_calendar=True, use_temp=False, use_apparent_temp=False,
            use_rrp=False, demand_lags=[], rolling_windows=[],
        ),
        model_factory=lambda: NumpyOLS(alpha=1.0),
    )


def _ols_calendar_temp() -> Candidate:
    return Candidate(
        name="ols_calendar_temp",
        feature_config=FeatureConfig(
            use_calendar=True, use_temp=True, use_apparent_temp=False,
            use_rrp=False, demand_lags=[], rolling_windows=[],
        ),
        model_factory=lambda: NumpyOLS(alpha=1.0),
    )


def _ols_calendar_temp_apptemp() -> Candidate:
    return Candidate(
        name="ols_calendar_temp_apptemp",
        feature_config=FeatureConfig(
            use_calendar=True, use_temp=True, use_apparent_temp=True,
            use_rrp=False, demand_lags=[], rolling_windows=[],
        ),
        model_factory=lambda: NumpyOLS(alpha=1.0),
    )


def _ols_full() -> Candidate:
    return Candidate(
        name="ols_full",
        feature_config=FeatureConfig(
            use_calendar=True, use_temp=True, use_apparent_temp=True,
            use_rrp=True, demand_lags=[1, 7], rolling_windows=[7],
        ),
        model_factory=lambda: NumpyOLS(alpha=1.0),
    )


EXPERIMENTS = [
    ("E1_baselines_only", []),
    ("E2_add_ols_calendar", [_ols_calendar()]),
    ("E3_add_ols_temp", [_ols_calendar(), _ols_calendar_temp()]),
    ("E4_add_apparent_temp", [_ols_calendar(), _ols_calendar_temp(), _ols_calendar_temp_apptemp()]),
    ("E5_full_stack", [_ols_calendar(), _ols_calendar_temp(),
                       _ols_calendar_temp_apptemp(), _ols_full()]),
]


def main() -> None:
    df = load_merged()
    warnings = validate_merged(df)
    if warnings:
        print("Data quality warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()
    splits = make_splits(df)
    # Dry runs land in their own subdirectory so the master log + champion
    # for this controlled batch are self-contained and easy to inspect.
    results_dir = Path("experiments/results/dry_runs")
    results_dir.mkdir(parents=True, exist_ok=True)

    for label, extras in EXPERIMENTS:
        print()
        print("#" * 70)
        print(f"# {label}")
        print("#" * 70)
        candidates = baseline_candidates() + extras
        run_loop(splits, candidates=candidates, results_dir=results_dir)


if __name__ == "__main__":
    main()
