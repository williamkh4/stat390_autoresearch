"""
Run the AutoResearch loop ONCE.

What happens in one run:
  1. Load + merge the Kaggle Victoria and Open-Meteo CSVs.
  2. Carve off the locked test set (final 365 days) and validation slice
     (180 days before test). Test is never touched here.
  3. Fit every registered candidate on the train split; score MSE on val.
  4. Track wall-clock runtime per candidate + total, count fits as budget,
     and write:
       * experiments/results/run_<id>.json         (full report)
       * experiments/results/run_<id>_leaderboard.csv

Usage:
    python run_autoresearch.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time

from src.autoresearch import default_candidates, run_loop
from src.data_loader import load_merged
from src.split import make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AutoResearch iteration.")
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="Where to write run_<id>.json and the leaderboard CSV.",
    )
    args = parser.parse_args()

    wall_start = time.perf_counter()

    df = load_merged()
    splits = make_splits(df)

    print("=" * 60)
    print("AutoResearch loop: one iteration")
    print("=" * 60)
    print(splits.describe())
    print()

    candidates = default_candidates()
    print(f"Candidates this run ({len(candidates)}):")
    for c in candidates:
        print(f"  - {c.describe()}")
    print()

    report = run_loop(splits, candidates=candidates, results_dir=Path(args.results_dir))

    total_wall = time.perf_counter() - wall_start
    print()
    print("=" * 60)
    print("Budget summary")
    print("=" * 60)
    print(f"  wall clock (including data load): {total_wall:.2f} s")
    print(f"  loop-only runtime:                {report.total_runtime_sec:.2f} s")
    print(f"  model fits (budget units):        {report.fit_count}")
    print(f"  avg runtime per fit:              "
          f"{report.total_runtime_sec / max(report.fit_count, 1):.3f} s")
    print(f"  run_id:                           {report.run_id}")


if __name__ == "__main__":
    main()
