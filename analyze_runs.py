"""
Cross-run analysis over the master log.

Iter-2 surfaces walk-forward `mean ± std` per candidate. When the master
log has no walk-forward columns (legacy iter-1 history), the script falls
back to the deterministic point estimates and `std = 0`.

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


def _has_mean_cols(df: pd.DataFrame) -> bool:
    return f"{PRIMARY_METRIC_NAME}_mean" in df.columns and df[f"{PRIMARY_METRIC_NAME}_mean"].notna().any()


def _format_mean_std(row: pd.Series, key: str) -> str:
    mean = row.get(f"{key}_mean")
    std = row.get(f"{key}_std")
    if pd.notna(mean) and pd.notna(std):
        return f"{mean:>14,.2f} ± {std:>12,.2f}"
    if pd.notna(row.get(key)):
        return f"{row[key]:>14,.2f}"
    return f"{'—':>14}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-run analysis.")
    parser.add_argument("--results-dir", default="experiments/auto_runs")
    parser.add_argument("--top", type=int, default=10,
                        help="How many leaderboard rows to print.")
    parser.add_argument("--ablation", action="store_true",
                        help="Print mean, best, and across-run variance per candidate.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    log_path = results_dir / MASTER_LOG_NAME
    champ_path = results_dir / CHAMPION_NAME

    if not log_path.exists():
        raise SystemExit(
            f"No master log at {log_path}. Run python run_autoresearch.py first."
        )

    df = pd.read_csv(log_path)
    df = df[df["error"].fillna("") == ""]
    walk_forward = _has_mean_cols(df)
    rank_col = f"{PRIMARY_METRIC_NAME}_mean" if walk_forward else PRIMARY_METRIC_NAME

    # ---- Champion ---------------------------------------------------------
    print("=" * 70)
    print("Current champion")
    print("=" * 70)
    if champ_path.exists():
        champ = json.loads(champ_path.read_text())
        print(f"  name           : {champ['name']}")
        m = champ.get("metric_mean") or champ.get("metric")
        s = champ.get("metric_std")
        si = champ.get("metric_std_indep")
        nf = champ.get("n_folds")
        proto = champ.get("protocol", "?")
        if s is not None:
            print(f"  {PRIMARY_METRIC_NAME:<14} : {m:,.2f} ± {s:,.2f}  "
                  f"(std_indep {si:,.2f}, n_folds={nf}, protocol={proto})")
        else:
            print(f"  {PRIMARY_METRIC_NAME:<14} : {m:,.2f}  (legacy point estimate)")
        print(f"  set in run     : {champ.get('run_id', '?')}")
        print(f"  uses_predicted_rrp : {champ.get('uses_predicted_rrp', False)}")
        print(f"  uses_observed_rrp  : {champ.get('uses_observed_rrp', False)}")
    else:
        print("  (no champion yet)")

    # ---- All-time leaderboard --------------------------------------------
    print()
    print("=" * 70)
    print(f"All-time leaderboard (best run per candidate)  top {args.top}  "
          f"({'walk-forward mean ± std' if walk_forward else 'iter-1 point estimate'})")
    print("=" * 70)
    best_per = (
        df.sort_values(rank_col)
          .groupby("candidate_name", as_index=False)
          .first()
          .sort_values(rank_col)
          .head(args.top)
    )
    if walk_forward:
        for _, row in best_per.iterrows():
            mse = _format_mean_std(row, PRIMARY_METRIC_NAME)
            indep = row.get(f"{PRIMARY_METRIC_NAME}_std_indep")
            indep_str = f"std_indep={indep:,.0f}" if pd.notna(indep) else "std_indep=—"
            nf = int(row.get("n_folds") or 0)
            prrp = bool(row.get("uses_predicted_rrp", False))
            tag = " [predRRP]" if prrp else ""
            print(f"  {mse}  {indep_str}  n_folds={nf}  "
                  f"run_id={row.get('run_id', '?')}  {row['candidate_name']}{tag}")
    else:
        cols = ["candidate_name", PRIMARY_METRIC_NAME, "rmse_demand", "mae_demand",
                "runtime_sec", "n_features", "run_id"]
        print(best_per[cols].to_string(index=False))

    # ---- Per-run summary --------------------------------------------------
    print()
    print("=" * 70)
    print("Per-run summary")
    print("=" * 70)
    per_run = (
        df.groupby("run_id")
          .agg(
              candidates=("candidate_name", "count"),
              best_metric=(rank_col, "min"),
              best_candidate=("candidate_name",
                              lambda s: df.loc[s.index, [rank_col, "candidate_name"]]
                                        .sort_values(rank_col)
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
        print("=" * 70)
        print("Ablation: per-candidate stats across all runs")
        print("=" * 70)
        agg_kwargs = {
            "n_runs": (rank_col, "count"),
            "best": (rank_col, "min"),
            "mean_across_runs": (rank_col, "mean"),
            "std_across_runs": (rank_col, "std"),
        }
        if walk_forward:
            agg_kwargs.update({
                "fold_std_typical": (f"{PRIMARY_METRIC_NAME}_std", "mean"),
                "fold_std_indep_typical": (f"{PRIMARY_METRIC_NAME}_std_indep", "mean"),
                "n_folds": ("n_folds", "first"),
            })
        ablation = (
            df.groupby("candidate_name")
              .agg(**agg_kwargs)
              .sort_values("best")
        )
        print(ablation.to_string())

        if walk_forward:
            # std_indep should typically be >= std (independent folds spread wider).
            n_violations = int(
                (ablation["fold_std_indep_typical"]
                 < ablation["fold_std_typical"]).fillna(False).sum()
            )
            print()
            print(f"Note: std_indep < std for {n_violations} / {len(ablation)} candidates. "
                  f"Expect std_indep >= std on average; large violations suggest "
                  f"the 5 non-overlapping folds happened to be quieter than the "
                  f"overlapping set on that candidate.")


if __name__ == "__main__":
    main()
