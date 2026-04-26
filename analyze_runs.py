"""
Cross-run analysis over the master log.

By default, prints the all-time leaderboard (best result for each candidate
name across every run), the current champion, and a per-run summary.

Use --ablation to see same-name candidates compared head-to-head: helpful
for "did adding apparent_temp lower MSE?" style questions.

Usage:
    python analyze_runs.py
    python analyze_runs.py --top 10
    python analyze_runs.py --ablation
    python analyze_runs.py --results-dir path/to/results
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd

from src.autoresearch import CHAMPION_NAME, MASTER_LOG_NAME
from src.metrics import PRIMARY_METRIC_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-run analysis.")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--top", type=int, default=10,
                        help="How many leaderboard rows to print.")
    parser.add_argument("--ablation", action="store_true",
                        help="Print mean and best per candidate across runs.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    log_path = results_dir / MASTER_LOG_NAME
    champ_path = results_dir / CHAMPION_NAME

    if not log_path.exists():
        raise SystemExit(
            f"No master log at {log_path}. Run python run_autoresearch.py first."
        )

    df = pd.read_csv(log_path)
    df = df[df["error"].fillna("") == ""]   # drop failed candidates

    # ---- Champion ---------------------------------------------------------
    print("=" * 60)
    print("Current champion")
    print("=" * 60)
    if champ_path.exists():
        champ = json.loads(champ_path.read_text())
        print(f"  name        : {champ['name']}")
        print(f"  {PRIMARY_METRIC_NAME:<12}: {champ['metric']:,.2f}")
        print(f"  set in run  : {champ.get('run_id', '?')}")
        print(f"  set at      : {champ.get('timestamp_utc', '?')}")
    else:
        print("  (no champion yet)")

    # ---- All-time leaderboard --------------------------------------------
    print()
    print("=" * 60)
    print(f"All-time leaderboard (best run per candidate)  top {args.top}")
    print("=" * 60)
    best_per = (
        df.sort_values(PRIMARY_METRIC_NAME)
          .groupby("candidate_name", as_index=False)
          .first()
          .sort_values(PRIMARY_METRIC_NAME)
          .head(args.top)
    )
    cols = ["candidate_name", PRIMARY_METRIC_NAME, "rmse_demand", "mae_demand",
            "runtime_sec", "n_features", "run_id"]
    print(best_per[cols].to_string(index=False))

    # ---- Per-run summary --------------------------------------------------
    print()
    print("=" * 60)
    print("Per-run summary")
    print("=" * 60)
    per_run = (
        df.groupby("run_id")
          .agg(
              candidates=("candidate_name", "count"),
              best_metric=(PRIMARY_METRIC_NAME, "min"),
              best_candidate=("candidate_name",
                              lambda s: df.loc[s.index, [PRIMARY_METRIC_NAME, "candidate_name"]]
                                        .sort_values(PRIMARY_METRIC_NAME)
                                        .iloc[0]["candidate_name"]),
              total_runtime=("runtime_sec", "sum"),
              timestamp=("timestamp_utc", "first"),
          )
          .sort_values("timestamp")
    )
    print(per_run.to_string())

    # ---- Ablation table ---------------------------------------------------
    if args.ablation:
        print()
        print("=" * 60)
        print("Ablation: per-candidate stats across all runs")
        print("=" * 60)
        ablation = (
            df.groupby("candidate_name")
              .agg(
                  n_runs=(PRIMARY_METRIC_NAME, "count"),
                  best=(PRIMARY_METRIC_NAME, "min"),
                  mean=(PRIMARY_METRIC_NAME, "mean"),
                  std=(PRIMARY_METRIC_NAME, "std"),
              )
              .sort_values("best")
        )
        print(ablation.to_string())


if __name__ == "__main__":
    main()
