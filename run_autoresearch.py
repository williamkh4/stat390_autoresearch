"""
Run the AutoResearch loop ONCE -- self-driving.

What happens in one run:
  1. Load + merge the Kaggle Victoria and Open-Meteo CSVs.
  2. Carve off the locked test set (final 365 days) and validation slice
     (180 days before test). Test is never touched here.
  3. Read the current champion from `experiments/auto_runs/champion.json`
     (None on first run) and the cross-run history from `master_log.csv`.
  4. Auto-generate candidates via `auto_candidates()`:
       * Both seasonal-naive baselines (always)
       * The champion's exact config (always, so noise is visible)
       * N "challengers" -- one-knob mutations of the champion's feature
         config first, then random untried points from the search space
  5. Fit + score every candidate on validation, write the leaderboard,
     append rows to `master_log.csv`, auto-promote the champion if any
     challenger beat it.

Because step 3 reads history and step 4 dedupes against it, every run does
*new* work without you editing any code. To enlarge or shrink the per-run
search effort, pass `--n-challengers`.

Usage:
    python run_autoresearch.py
    python run_autoresearch.py --n-challengers 6
    python run_autoresearch.py --seed 42                  # reproducible draw
    python run_autoresearch.py --results-dir my/path/here
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time

from src.autoresearch import (
    CHAMPION_NAME,
    auto_candidates,
    _load_champion,
    run_loop,
)
from src.data_loader import load_merged
from src.split import make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AutoResearch iteration (self-driving).")
    parser.add_argument(
        "--results-dir",
        default="experiments/auto_runs",
        help="Where to write run_<id>.json, leaderboards, master_log.csv, champion.json.",
    )
    parser.add_argument(
        "--n-challengers",
        type=int,
        default=4,
        help="How many non-baseline, non-champion candidates to try this run (default 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional rng seed for the challenger draw. If omitted, the seed "
            "is derived from the number of rows in master_log.csv so each new "
            "run automatically pulls a different slice of the search space."
        ),
    )
    parser.add_argument(
        "--evaluate-on-test",
        action="store_true",
        help=(
            "Score every candidate on the locked test set IN ADDITION to "
            "validation. Adds *_test columns to master_log and the run JSON. "
            "Champion promotion still uses validation MSE unless "
            "--promote-on=test is also passed. Use this flag to monitor the "
            "val->test gap per candidate without burning the test set."
        ),
    )
    parser.add_argument(
        "--promote-on",
        choices=["val", "test"],
        default="val",
        help=(
            "Which metric source decides champion promotion. Default 'val' "
            "preserves the locked-test-set design. Setting this to 'test' "
            "requires --evaluate-on-test and means the loop will use TEST "
            "MSE to rank and promote -- the loop prints a multi-line warning "
            "in this mode because every candidate fit becomes one more 'look' "
            "at the test set."
        ),
    )
    args = parser.parse_args()
    if args.promote_on == "test" and not args.evaluate_on_test:
        parser.error("--promote-on=test requires --evaluate-on-test.")

    wall_start = time.perf_counter()

    df = load_merged()
    splits = make_splits(df)

    print("=" * 60)
    print("AutoResearch loop: one iteration (self-driving)")
    print("=" * 60)
    print(splits.describe())
    print()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    prev_champion = _load_champion(results_dir / CHAMPION_NAME)

    candidates = auto_candidates(
        prev_champion=prev_champion,
        results_dir=results_dir,
        n_challengers=args.n_challengers,
        seed=args.seed,
    )

    print(f"Candidates this run ({len(candidates)}):")
    for c in candidates:
        tag = "baseline" if c.is_baseline else (
            "champion" if prev_champion and c.name == prev_champion.get("name") else "challenger"
        )
        print(f"  - [{tag}] {c.describe()}")
    print()

    report = run_loop(
        splits, candidates=candidates, results_dir=results_dir,
        evaluate_on_test=args.evaluate_on_test,
        promote_on=args.promote_on,
    )

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
