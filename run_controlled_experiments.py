"""
Controlled experiment runner.

The auto-driving loop in `run_autoresearch.py` is good at *exploring* the
search space, but its choices are stochastic (random untried + champion
mutations). For *interpretation* we sometimes need a deliberate sweep where
exactly one variable changes per experiment and everything else is held
fixed. This script runs that sweep and writes its artifacts to
`experiments/controlled/` so they don't pollute the main `auto_runs/`
master log.

Three sweep series (13 experiments, 2 of which are duplicates -- C0 = A4 = B4
== the current champion -- so 11 distinct fits):

  Series A -- Feature ablation, model fixed at champion's MLP spec.
              A1 cal+lag1-7
              A2 cal+temp+lag1-7
              A3 cal+temp+apptemp+lag1-7
              A4 cal+temp+apptemp+lag1-7+roll7      (champion features)
              A5 full (cal+temp+apptemp+RRP+lag1-7+roll7)

  Series B -- Model ablation, features fixed at champion's preset.
              B1 numpy_ols  (alpha=1)
              B2 ridge      (alpha=1)
              B3 rf         (n=200, depth=None, leaf=5, sqrt)
              B4 mlp        (champion = (128,64), relu, adam, alpha=0.001, lr=0.01)
              B5 gbm        (n=300, depth=3, lr=0.05)

  Series C -- MLP hyperparameter sweep on champion features. Vary lr only.
              C1 lr=0.001
              C2 lr=0.005
              C3 lr=0.01    (champion)
              C4 lr=0.02
              C5 lr=0.05

Every artifact goes into `experiments/controlled/`:
  controlled_results.csv   one row per experiment (series, label, MSE, runtime)
  series_<X>.json          per-series detail
  controlled_master.csv    same shape as master_log.csv for re-using analyze tools

Usage:
    python run_controlled_experiments.py
    python run_controlled_experiments.py --results-dir experiments/controlled
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple
import argparse
import json

import pandas as pd

from src.autoresearch import (
    Candidate,
    _make_factory,
    score_candidate,
)
from src.data_loader import load_merged
from src.features import FeatureConfig
from src.metrics import PRIMARY_METRIC_NAME
from src.split import make_splits


# Champion fixtures used by all three series. Keep these in lockstep with
# the current champion in experiments/auto_runs/champion.json. If the
# champion changes, update these constants and re-run; the controlled set is
# meant to ablate around that champion.
CHAMPION_FEATURE_CONFIG = FeatureConfig(
    use_calendar=True, use_temp=True, use_apparent_temp=True, use_rrp=False,
    demand_lags=[1, 7], rolling_windows=[7],
)
CHAMPION_MLP_KWARGS = (
    ("hidden_layer_sizes", (128, 64)),
    ("activation", "relu"),
    ("solver", "adam"),
    ("alpha", 0.001),
    ("learning_rate_init", 0.01),
    ("max_iter", 500),
    ("random_state", 0),
)


def fc(use_calendar=False, use_temp=False, use_apparent_temp=False,
       use_rrp=False, demand_lags=None, rolling_windows=None) -> FeatureConfig:
    return FeatureConfig(
        use_calendar=use_calendar, use_temp=use_temp,
        use_apparent_temp=use_apparent_temp, use_rrp=use_rrp,
        demand_lags=list(demand_lags or []),
        rolling_windows=list(rolling_windows or []),
    )


def series_a() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """Feature ablation. Model held fixed at champion MLP."""
    base = ("mlp", CHAMPION_MLP_KWARGS)
    return [
        ("A", "A1_cal_lag",          fc(use_calendar=True, demand_lags=[1, 7]),                                                   *base),
        ("A", "A2_cal_temp_lag",     fc(use_calendar=True, use_temp=True, demand_lags=[1, 7]),                                    *base),
        ("A", "A3_+apparent_temp",   fc(use_calendar=True, use_temp=True, use_apparent_temp=True, demand_lags=[1, 7]),            *base),
        ("A", "A4_+rolling7",        CHAMPION_FEATURE_CONFIG,                                                                     *base),
        ("A", "A5_+RRP_full",        fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                                        use_rrp=True, demand_lags=[1, 7], rolling_windows=[7]),                                   *base),
    ]


def series_b() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """Model class ablation. Features held fixed at champion preset."""
    f = CHAMPION_FEATURE_CONFIG
    return [
        ("B", "B1_numpy_ols", f, "numpy_ols", (("alpha", 1.0),)),
        ("B", "B2_ridge",     f, "ridge",     (("alpha", 1.0), ("random_state", 0))),
        ("B", "B3_rf",        f, "rf",        (("n_estimators", 200), ("max_depth", None),
                                               ("min_samples_leaf", 5), ("max_features", "sqrt"),
                                               ("random_state", 0), ("n_jobs", -1))),
        ("B", "B4_mlp_champ", f, "mlp",       CHAMPION_MLP_KWARGS),
        ("B", "B5_gbm",       f, "gbm",       (("n_estimators", 300), ("max_depth", 3),
                                               ("learning_rate", 0.05), ("subsample", 1.0),
                                               ("random_state", 0))),
    ]


def series_c() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """MLP learning_rate_init sweep on champion features."""
    f = CHAMPION_FEATURE_CONFIG
    base_kw = list(CHAMPION_MLP_KWARGS)

    def with_lr(lr: float) -> tuple:
        out = []
        for k, v in base_kw:
            out.append((k, lr) if k == "learning_rate_init" else (k, v))
        return tuple(out)

    return [
        ("C", "C1_lr0.001", f, "mlp", with_lr(0.001)),
        ("C", "C2_lr0.005", f, "mlp", with_lr(0.005)),
        ("C", "C3_lr0.01",  f, "mlp", with_lr(0.01)),    # = champion
        ("C", "C4_lr0.02",  f, "mlp", with_lr(0.02)),
        ("C", "C5_lr0.05",  f, "mlp", with_lr(0.05)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled experiments.")
    parser.add_argument("--results-dir", default="experiments/controlled")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    splits = make_splits(df)
    print(splits.describe())
    print()

    all_specs = series_a() + series_b() + series_c()
    print(f"Running {len(all_specs)} controlled experiments")
    print(f"  series A (feature ablation): {sum(1 for s in all_specs if s[0]=='A')} experiments")
    print(f"  series B (model class):      {sum(1 for s in all_specs if s[0]=='B')} experiments")
    print(f"  series C (MLP lr sweep):     {sum(1 for s in all_specs if s[0]=='C')} experiments")
    print()

    rows = []
    master_rows = []
    timestamp = pd.Timestamp.utcnow().isoformat()
    for series, label, fcfg, model_type, kwargs in all_specs:
        cand = Candidate(
            name=f"controlled_{label}",
            feature_config=fcfg,
            model_factory=_make_factory(model_type, kwargs),
            is_baseline=False,
        )
        print(f"[{series}] {label}  ({model_type})  features[{fcfg.describe()}]")
        res = score_candidate(cand, splits)
        if res.error:
            print(f"  ! ERROR: {res.error}  ({res.runtime_sec:.2f}s)")
        else:
            print(f"  {PRIMARY_METRIC_NAME}={res.metrics[PRIMARY_METRIC_NAME]:.0f} "
                  f"({res.runtime_sec:.2f}s, n_features={res.n_features})")
        rows.append({
            "series": series,
            "label": label,
            "model_type": model_type,
            "feature_config": fcfg.describe(),
            "n_features": res.n_features,
            PRIMARY_METRIC_NAME: res.metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand": res.metrics.get("rmse_demand"),
            "mae_demand": res.metrics.get("mae_demand"),
            "runtime_sec": res.runtime_sec,
            "error": res.error or "",
        })
        master_rows.append({
            "run_id": "controlled",
            "timestamp_utc": timestamp,
            "candidate_name": f"controlled_{label}",
            "is_baseline": False,
            PRIMARY_METRIC_NAME: res.metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand": res.metrics.get("rmse_demand"),
            "mae_demand": res.metrics.get("mae_demand"),
            "runtime_sec": res.runtime_sec,
            "n_features": res.n_features,
            "n_train": res.n_train,
            "n_val": res.n_val,
            "error": res.error or "",
        })

    out_csv = results_dir / "controlled_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print()
    print(f"Wrote: {out_csv}")

    # Drop a master_log-shaped CSV alongside so analyze_runs.py works on it
    # if someone points it at this directory.
    master_csv = results_dir / "controlled_master.csv"
    pd.DataFrame(master_rows).to_csv(master_csv, index=False)
    print(f"Wrote: {master_csv}")

    # Per-series JSON for archival inspection.
    for s in ("A", "B", "C"):
        s_rows = [r for r in rows if r["series"] == s]
        path = results_dir / f"series_{s}.json"
        json.dump({"series": s, "results": s_rows},
                  open(path, "w"), indent=2, default=str)
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
