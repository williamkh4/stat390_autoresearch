"""
Baseline analysis script.

Computes the locked validation metric (MSE of demand) for every seasonal-naive
baseline registered in the AutoResearch candidate list, and writes a small
JSON + CSV report alongside the AutoResearch run artifacts.

Output (relative to the current working directory):
    experiments/results/baseline_<timestamp>.json
    experiments/results/baseline_<timestamp>.csv

Usage:
    python run_baseline.py
    python run_baseline.py --results-dir path/to/results
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time

import pandas as pd

from src.autoresearch import baseline_candidates, score_candidate
from src.data_loader import load_merged
from src.metrics import PRIMARY_METRIC_NAME
from src.split import make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline analysis.")
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="Where to write baseline_<timestamp>.json + .csv.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    df = load_merged()
    splits = make_splits(df)

    print("=" * 60)
    print("Baseline analysis")
    print("=" * 60)
    print(splits.describe())
    print()

    # Pull baselines from the shared registry -- same ones the AutoResearch
    # loop uses, so "the number to beat" stays consistent across scripts.
    baselines = baseline_candidates()
    if not baselines:
        raise SystemExit("No baseline candidates registered.")

    results = []
    for cand in baselines:
        print(f"Scoring: {cand.name}")
        res = score_candidate(cand, splits)
        if res.error:
            print(f"  ! error: {res.error}")
        else:
            print(f"  {PRIMARY_METRIC_NAME} = {res.metrics[PRIMARY_METRIC_NAME]:>14,.2f}")
            print(f"  rmse_demand = {res.metrics['rmse_demand']:>14,.2f}")
            print(f"  mae_demand  = {res.metrics['mae_demand']:>14,.2f}")
        results.append(res)
    runtime = time.perf_counter() - t0

    # Identify the strongest baseline -- this is the "number to beat".
    ok_results = [r for r in results if not r.error]
    best = min(ok_results, key=lambda r: r.metrics[PRIMARY_METRIC_NAME]) if ok_results else None

    y_true = splits.val["demand"].to_numpy(dtype=float)
    print()
    print(f"Validation demand: mean = {y_true.mean():,.2f}, "
          f"std = {y_true.std():,.2f}, n = {len(y_true)}")
    if best:
        print(f"Strongest baseline:  {best.name}  "
              f"{PRIMARY_METRIC_NAME} = {best.metrics[PRIMARY_METRIC_NAME]:,.2f}")
    print(f"Runtime: {runtime:.3f} s")
    print("Test set is LOCKED and has not been touched.")

    # Write artifacts.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")

    report = {
        "kind": "baseline_analysis",
        "timestamp_utc": stamp,
        "splits_summary": splits.describe(),
        "runtime_sec": runtime,
        "validation_demand_stats": {
            "mean": float(y_true.mean()),
            "std": float(y_true.std()),
            "n": int(len(y_true)),
        },
        "best_baseline": best.name if best else None,
        "best_metric": best.metrics[PRIMARY_METRIC_NAME] if best else None,
        "results": [asdict(r) for r in results],
    }
    json_path = results_dir / f"baseline_{stamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    csv_path = results_dir / f"baseline_{stamp}.csv"
    pd.DataFrame([
        {
            "name": r.name,
            PRIMARY_METRIC_NAME: r.metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand": r.metrics.get("rmse_demand"),
            "mae_demand": r.metrics.get("mae_demand"),
            "runtime_sec": r.runtime_sec,
            "error": r.error or "",
        }
        for r in results
    ]).sort_values(PRIMARY_METRIC_NAME).to_csv(csv_path, index=False)

    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
