"""
Targeted stress test for the MLP and GBM model frameworks.

Where `run_controlled_experiments.py` ablates a single project champion
(features, model class, MLP learning rate), this script drills into the
two best-performing model frameworks and varies the levers that matter
*within each framework's own context*.

Series design (4 per framework, 5 experiments each = 40 fits):

  MLP -- pivots on the champion MLP spec, champion features.
    M1  Feature selection           (calendar/temp/apptemp/roll7/RRP)
    M2  Architecture                (hidden_layer_sizes from (16,) to (256,128,64))
    M3  L2 regularization           (alpha from 1e-5 to 1e-1)
    M4  learning_rate_init          (0.001 -> 0.05; revisits Series C in the
                                     controlled bundle, on purpose, so all four
                                     MLP knobs sit in one comparable plot)

  GBM -- pivots on a sensible GBM default that already shows up in the auto
         search space (n=300, depth=3, lr=0.05) on the champion features.
    G1  Feature selection           (same five levels as M1)
    G2  Number of estimators        (100 -> 1000)
    G3  Tree depth                  (2 -> 8)
    G4  Learning rate (shrinkage)   (0.01 -> 0.2)

Each candidate is fit on train and scored on *both* val and test. Both
seasonal-naive baselines run first so the bundle is self-contained for
analysis (validation + test reference points come along with the rest of
the rows).

Outputs (under `experiments/stress_test/`):
    stress_results.csv      one row per experiment
    stress_master.csv       master_log-shaped CSV for re-using analyze tools
    series_<X>.json         per-series detail (M1, M2, M3, M4, G1, G2, G3, G4)

Usage:
    python run_stress_test.py
    python run_stress_test.py --results-dir other/dir
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import argparse
import json

import pandas as pd

from src.autoresearch import (
    Candidate,
    _make_factory,
    baseline_candidates,
    score_candidate,
)
from src.data_loader import load_merged
from src.features import FeatureConfig
from src.metrics import PRIMARY_METRIC_NAME
from src.split import make_splits


# ---- Pivots: keep aligned with experiments/auto_runs/champion.json ---------

CHAMPION_FEATURES = FeatureConfig(
    use_calendar=True, use_temp=True, use_apparent_temp=True, use_rrp=False,
    demand_lags=[1, 7], rolling_windows=[7],
)

# MLP champion spec (matches champion.json at the time of writing).
MLP_BASE = (
    ("hidden_layer_sizes", (128, 64)),
    ("activation", "relu"),
    ("solver", "adam"),
    ("alpha", 0.001),
    ("learning_rate_init", 0.01),
    ("max_iter", 500),
    ("random_state", 0),
)

# GBM "good default" -- same shape as the strongest GBM that has surfaced
# in the auto-loop search space; we sweep around it so each axis is a
# clean one-knob ablation.
GBM_BASE = (
    ("n_estimators", 300),
    ("max_depth", 3),
    ("learning_rate", 0.05),
    ("subsample", 1.0),
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


def _set_kwarg(base: tuple, key: str, value) -> tuple:
    """Return base with `key` overwritten (preserving order) or appended."""
    out: List[Tuple[str, object]] = []
    seen = False
    for k, v in base:
        if k == key:
            out.append((k, value))
            seen = True
        else:
            out.append((k, v))
    if not seen:
        out.append((key, value))
    return tuple(out)


# ---- MLP series -----------------------------------------------------------

FEATURE_LADDER = [
    ("cal_lag",            fc(use_calendar=True, demand_lags=[1, 7])),
    ("cal_temp_lag",       fc(use_calendar=True, use_temp=True, demand_lags=[1, 7])),
    ("cal_temp_apt_lag",   fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                              demand_lags=[1, 7])),
    ("champion_features",  CHAMPION_FEATURES),                   # ★ champion preset
    ("full_with_RRP",      fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                              use_rrp=True, demand_lags=[1, 7], rolling_windows=[7])),
]


def series_m1() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """MLP feature selection. Model held fixed at champion MLP."""
    out = []
    for tag, fcfg in FEATURE_LADDER:
        out.append(("M1", f"M1_{tag}", fcfg, "mlp", MLP_BASE))
    return out


def series_m2() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """MLP architecture sweep (hidden_layer_sizes). Features fixed at champion."""
    architectures = [
        ("hls16",        (16,)),
        ("hls32x16",     (32, 16)),
        ("hls64x32",     (64, 32)),
        ("hls128x64",    (128, 64)),       # ★ champion arch
        ("hls256x128x64",(256, 128, 64)),
    ]
    out = []
    for tag, hls in architectures:
        kw = _set_kwarg(MLP_BASE, "hidden_layer_sizes", hls)
        out.append(("M2", f"M2_{tag}", CHAMPION_FEATURES, "mlp", kw))
    return out


def series_m3() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """MLP L2-regularization (alpha) sweep. Features + arch fixed."""
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # 1e-3 = champion
    out = []
    for a in alphas:
        kw = _set_kwarg(MLP_BASE, "alpha", a)
        tag = f"alpha{a:g}"
        out.append(("M3", f"M3_{tag}", CHAMPION_FEATURES, "mlp", kw))
    return out


def series_m4() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """MLP learning_rate_init sweep on champion features + arch."""
    lrs = [0.001, 0.005, 0.01, 0.02, 0.05]    # 0.01 = champion
    out = []
    for lr in lrs:
        kw = _set_kwarg(MLP_BASE, "learning_rate_init", lr)
        out.append(("M4", f"M4_lr{lr:g}", CHAMPION_FEATURES, "mlp", kw))
    return out


# ---- GBM series -----------------------------------------------------------

def series_g1() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """GBM feature selection. Model held fixed at GBM_BASE."""
    out = []
    for tag, fcfg in FEATURE_LADDER:
        out.append(("G1", f"G1_{tag}", fcfg, "gbm", GBM_BASE))
    return out


def series_g2() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """GBM n_estimators sweep. Features fixed."""
    ns = [100, 200, 300, 500, 1000]            # 300 = default
    out = []
    for n in ns:
        kw = _set_kwarg(GBM_BASE, "n_estimators", n)
        out.append(("G2", f"G2_n{n}", CHAMPION_FEATURES, "gbm", kw))
    return out


def series_g3() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """GBM tree depth sweep. Features fixed."""
    depths = [2, 3, 4, 6, 8]                   # 3 = default
    out = []
    for d in depths:
        kw = _set_kwarg(GBM_BASE, "max_depth", d)
        out.append(("G3", f"G3_depth{d}", CHAMPION_FEATURES, "gbm", kw))
    return out


def series_g4() -> List[Tuple[str, str, FeatureConfig, str, tuple]]:
    """GBM learning_rate (shrinkage) sweep. Features fixed."""
    lrs = [0.01, 0.03, 0.05, 0.1, 0.2]         # 0.05 = default
    out = []
    for lr in lrs:
        kw = _set_kwarg(GBM_BASE, "learning_rate", lr)
        out.append(("G4", f"G4_lr{lr:g}", CHAMPION_FEATURES, "gbm", kw))
    return out


# ---- Driver ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MLP / GBM stress test.")
    parser.add_argument("--results-dir", default="experiments/stress_test")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_merged()
    splits = make_splits(df)
    print(splits.describe())
    print()

    all_specs = (series_m1() + series_m2() + series_m3() + series_m4()
                 + series_g1() + series_g2() + series_g3() + series_g4())
    print(f"Running {len(all_specs)} stress-test experiments + 2 baselines")
    by_series: dict[str, int] = {}
    for s in all_specs:
        by_series[s[0]] = by_series.get(s[0], 0) + 1
    for s, n in by_series.items():
        print(f"  series {s}: {n} experiments")
    print()

    rows: List[dict] = []
    master_rows: List[dict] = []
    timestamp = pd.Timestamp.utcnow().isoformat()

    # Baselines first so the bundle is self-contained
    for cand in baseline_candidates():
        print(f"[baseline] {cand.name}")
        res = score_candidate(cand, splits, evaluate_on_test=True)
        if res.error:
            print(f"  ! ERROR: {res.error}  ({res.runtime_sec:.2f}s)")
        else:
            print(f"  val={res.metrics[PRIMARY_METRIC_NAME]:.0f}  "
                  f"test={res.test_metrics.get(PRIMARY_METRIC_NAME, float('nan')):.0f}  "
                  f"({res.runtime_sec:.2f}s)")
        row = {
            "series": "baseline",
            "label": cand.name,
            "model_type": "baseline",
            "feature_config": cand.feature_config.describe(),
            "n_features": res.n_features,
            PRIMARY_METRIC_NAME: res.metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand": res.metrics.get("rmse_demand"),
            "mae_demand": res.metrics.get("mae_demand"),
            "mse_demand_test": res.test_metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand_test": res.test_metrics.get("rmse_demand"),
            "mae_demand_test": res.test_metrics.get("mae_demand"),
            "runtime_sec": res.runtime_sec,
            "error": res.error or "",
        }
        rows.append(row)
        master_rows.append({**row, "run_id": "stress", "timestamp_utc": timestamp,
                            "candidate_name": cand.name, "is_baseline": True,
                            "n_train": res.n_train, "n_val": res.n_val,
                            "n_test": res.n_test})

    # All stress-test experiments
    for series, label, fcfg, model_type, kwargs in all_specs:
        cand = Candidate(
            name=f"stress_{label}",
            feature_config=fcfg,
            model_factory=_make_factory(model_type, kwargs),
            is_baseline=False,
        )
        print(f"[{series}] {label}  ({model_type})  features[{fcfg.describe()}]")
        res = score_candidate(cand, splits, evaluate_on_test=True)
        if res.error:
            print(f"  ! ERROR: {res.error}  ({res.runtime_sec:.2f}s)")
        else:
            print(f"  val={res.metrics[PRIMARY_METRIC_NAME]:.0f}  "
                  f"test={res.test_metrics.get(PRIMARY_METRIC_NAME, float('nan')):.0f}  "
                  f"({res.runtime_sec:.2f}s, n_features={res.n_features})")
        row = {
            "series": series,
            "label": label,
            "model_type": model_type,
            "feature_config": fcfg.describe(),
            "n_features": res.n_features,
            PRIMARY_METRIC_NAME: res.metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand": res.metrics.get("rmse_demand"),
            "mae_demand": res.metrics.get("mae_demand"),
            "mse_demand_test": res.test_metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand_test": res.test_metrics.get("rmse_demand"),
            "mae_demand_test": res.test_metrics.get("mae_demand"),
            "runtime_sec": res.runtime_sec,
            "error": res.error or "",
        }
        rows.append(row)
        master_rows.append({**row, "run_id": "stress", "timestamp_utc": timestamp,
                            "candidate_name": f"stress_{label}", "is_baseline": False,
                            "n_train": res.n_train, "n_val": res.n_val,
                            "n_test": res.n_test})

    out_csv = results_dir / "stress_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print()
    print(f"Wrote: {out_csv}")

    master_csv = results_dir / "stress_master.csv"
    pd.DataFrame(master_rows).to_csv(master_csv, index=False)
    print(f"Wrote: {master_csv}")

    for s in sorted({r["series"] for r in rows} - {"baseline"}):
        s_rows = [r for r in rows if r["series"] == s]
        path = results_dir / f"series_{s}.json"
        json.dump({"series": s, "results": s_rows},
                  open(path, "w"), indent=2, default=str)
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
