"""
Run the AutoResearch loop ONCE -- self-driving.

Iter-2 default protocol: walk-forward CV with ~10 folds (val_size=180,
step=90, min_train=730, expanding train, locked test held out). Pass
`--protocol holdout` to fall back to the iter-1 single-split protocol if
you need a fast smoke test or a direct iter-1 replay.

What happens in one run:
  1. Load + merge the Kaggle Victoria and Open-Meteo CSVs.
  2. Carve off the locked test set (final 365 days) and validation
     surface. For walk-forward this is ~10 (train, val) pairs; for
     holdout it's a single (train, val) split. The locked test set is
     never touched here.
  3. Read the current champion from `experiments/auto_runs/champion.json`
     (None on first run) and the cross-run history from `master_log.csv`.
  4. Auto-generate candidates via `auto_candidates()`:
       * Both seasonal-naive baselines (always)
       * The champion's exact config (always)
       * N "challengers" preferring primary preset × non-sanity-only,
         non-known_bad model specs, with one-knob mutations of the
         champion first
  5. Fit + score every candidate. Walk-forward mode aggregates
     mean ± std (and std_indep across the 5 non-overlapping folds) per
     metric. Champion promotion is noise-aware (iter-2 rule).
     Iter-2 columns are added to master_log.csv; legacy `mse_demand` is
     populated as the mean so downstream plots keep working.

Usage:
    python run_autoresearch.py                        # walk-forward, 4 challengers
    python run_autoresearch.py --n-challengers 6
    python run_autoresearch.py --seed 42              # reproducible draw
    python run_autoresearch.py --protocol holdout     # iter-1 single split
    python run_autoresearch.py --evaluate-on-test     # ALSO score on locked test
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
from src.split import make_splits, make_walk_forward_folds, walk_forward_summary


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
        "--protocol",
        choices=["walk_forward", "holdout"],
        default="walk_forward",
        help=(
            "Validation protocol. Iter-2 default is 'walk_forward' (~10 folds, "
            "mean ± std reporting, noise-aware champion promotion). Pass "
            "'holdout' for the iter-1 single 180-day val window (fast, but "
            "no noise estimate)."
        ),
    )
    parser.add_argument(
        "--evaluate-on-test",
        action="store_true",
        help=(
            "Score every candidate on the locked test set IN ADDITION to "
            "validation. Adds *_test columns to master_log and the run JSON. "
            "Champion promotion still uses val unless --promote-on=test."
        ),
    )
    parser.add_argument(
        "--promote-on",
        choices=["val", "test"],
        default="val",
        help=(
            "Which metric source decides champion promotion. Default 'val' "
            "preserves the locked-test-set design. Setting 'test' requires "
            "--evaluate-on-test; the loop prints a warning in that mode "
            "because every fit becomes one more 'look' at the test set."
        ),
    )
    args = parser.parse_args()
    if args.promote_on == "test" and not args.evaluate_on_test:
        parser.error("--promote-on=test requires --evaluate-on-test.")

    wall_start = time.perf_counter()

    df = load_merged()

    if args.protocol == "walk_forward":
        validation_surface = make_walk_forward_folds(df)
        summary = walk_forward_summary(validation_surface)
    else:
        validation_surface = make_splits(df)
        summary = validation_surface.describe()

    print("=" * 60)
    print(f"AutoResearch loop: one iteration (self-driving, protocol={args.protocol})")
    print("=" * 60)
    print(summary)
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
        validation_surface,
        candidates=candidates,
        results_dir=results_dir,
        evaluate_on_test=args.evaluate_on_test,
        promote_on=args.promote_on,
    )

    total_wall = time.perf_counter() - wall_start
    print()
    print("=" * 60)
    print("Budget summary")
    print("=" * 60)
    print(f"  protocol:                         {args.protocol}")
    print(f"  wall clock (including data load): {total_wall:.2f} s")
    print(f"  loop-only runtime:                {report.total_runtime_sec:.2f} s")
    print(f"  model fits (budget units):        {report.fit_count}")
    print(f"  avg runtime per fit:              "
          f"{report.total_runtime_sec / max(report.fit_count, 1):.3f} s")
    print(f"  run_id:                           {report.run_id}")


if __name__ == "__main__":
    main()
