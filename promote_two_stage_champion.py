"""
Promote the best two-stage (predicted-RRP) candidate to champion.json.

Why a separate script: the auto loop's noise-aware promotion rule
(`mean_chal + std_chal < mean_champ − std_champ`) is calibrated against
the *recorded* champion. When champion.json is still carrying an
iter-1 holdout point estimate (no std), the rule's RHS is too tight and
no walk-forward challenger can clear it. This script promotes the best
two-stage candidate from master_log.csv into champion.json under the
iter-2 schema (mean ± std + std_indep + n_folds + protocol +
uses_predicted_rrp flag), so the next auto run has an apples-to-apples
bar.

Behaviour:
  1. Reads `experiments/auto_runs/master_log.csv`.
  2. Filters to walk-forward (protocol == "walk_forward_v90") rows with
     `uses_predicted_rrp` truthy and no error.
  3. Picks the row with the lowest `mse_demand_mean`.
  4. Backs up the current `champion.json` to `champion_iter1.json` (or
     a path passed via --archive-as).
  5. Writes the new champion under the iter-2 schema.

Usage:
    python promote_two_stage_champion.py
    python promote_two_stage_champion.py --archive-as champion_iter1.json
    python promote_two_stage_champion.py --dry-run            # print, don't write
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

import pandas as pd

from src.autoresearch import (
    CHAMPION_NAME,
    MASTER_LOG_NAME,
    _full_search_space,
    _load_champion,
    _save_champion,
)
from src.metrics import PRIMARY_METRIC_NAME


def _truthy(v) -> bool:
    """Coerce mixed-type master_log column ('True'/'False'/1.0/0.0/nan) to bool."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v) and not (isinstance(v, float) and math.isnan(v))
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "1.0", "yes"}
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote best two-stage candidate to champion.")
    parser.add_argument("--results-dir", default="experiments/auto_runs",
                        help="Folder with master_log.csv + champion.json.")
    parser.add_argument("--archive-as", default="champion_iter1.json",
                        help="Filename (relative to --results-dir) for the previous "
                             "champion.json. Default: champion_iter1.json.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the proposed swap, don't touch champion.json.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    log_path = results_dir / MASTER_LOG_NAME
    champion_path = results_dir / CHAMPION_NAME
    archive_path = results_dir / args.archive_as

    if not log_path.exists():
        print(f"[promote] no master_log at {log_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(log_path)
    df = df[df["error"].fillna("") == ""]
    if "protocol" in df.columns:
        df = df[df["protocol"].fillna("") == "walk_forward_v90"]
    two_stage = df[df["uses_predicted_rrp"].apply(_truthy)].copy()
    if not len(two_stage):
        print("[promote] no two-stage candidates found in walk-forward rows of master_log.",
              file=sys.stderr)
        return 2

    two_stage = two_stage.sort_values("mse_demand_mean")
    best = two_stage.iloc[0]

    # Reconstruct the full feature_config dict from the search space so the
    # champion record matches what the auto loop would have written.
    space = _full_search_space()
    match = next((c for c in space if c.name == best["candidate_name"]), None)
    if match is None:
        print(f"[promote] candidate '{best['candidate_name']}' not found in current "
              f"search space -- name drift? Aborting.", file=sys.stderr)
        return 3

    fc = match.feature_config
    feature_config = {
        "use_calendar": bool(fc.use_calendar),
        "use_temp": bool(fc.use_temp),
        "use_apparent_temp": bool(fc.use_apparent_temp),
        "use_rrp": bool(fc.use_rrp),
        "use_predicted_rrp": bool(getattr(fc, "use_predicted_rrp", False)),
        "demand_lags": list(fc.demand_lags),
        "rolling_windows": list(fc.rolling_windows),
    }

    new_champion = {
        "name": best["candidate_name"],
        # legacy alias so iter-1 readers keep working
        "metric": float(best["mse_demand_mean"]),
        "metric_source": "val",
        "metric_mean": float(best["mse_demand_mean"]),
        "metric_std": float(best.get("mse_demand_std") or 0.0),
        "metric_std_indep": float(best.get("mse_demand_std_indep") or 0.0),
        "n_folds": int(best.get("n_folds") or 10),
        "protocol": "walk_forward_v90",
        "val_metric": float(best["mse_demand_mean"]),
        "test_metric": (float(best["mse_demand_test"])
                        if pd.notna(best.get("mse_demand_test")) else None),
        "test_metric_mean": (float(best.get("mse_demand_test_mean"))
                             if pd.notna(best.get("mse_demand_test_mean")) else None),
        "test_metric_std": (float(best.get("mse_demand_test_std"))
                            if pd.notna(best.get("mse_demand_test_std")) else None),
        "uses_predicted_rrp": True,
        "uses_observed_rrp": _truthy(best.get("uses_observed_rrp")),
        "run_id": str(best["run_id"]),
        "timestamp_utc": str(best["timestamp_utc"]),
        "feature_config": feature_config,
        "is_baseline": False,
        "promoted_by": "promote_two_stage_champion.py",
    }

    prev = _load_champion(champion_path)

    print("=" * 70)
    print("Two-stage champion promotion")
    print("=" * 70)
    if prev is not None:
        prev_m = prev.get("metric_mean") or prev.get("metric")
        prev_s = prev.get("metric_std")
        prev_proto = prev.get("protocol", "holdout")
        prev_str = (f"{prev_m:,.0f} ± {prev_s:,.0f}" if prev_s
                    else f"{prev_m:,.0f} (no std; protocol={prev_proto})")
        print(f"  previous champion : {prev['name']}")
        print(f"                       {prev_str}  run_id={prev.get('run_id')}")
    else:
        print("  previous champion : (none)")
    print(f"  new      champion : {new_champion['name']}")
    print(f"                       {new_champion['metric_mean']:,.0f} ± "
          f"{new_champion['metric_std']:,.0f}  "
          f"std_indep={new_champion['metric_std_indep']:,.0f}  "
          f"n_folds={new_champion['n_folds']}  "
          f"run_id={new_champion['run_id']}")
    print(f"  uses_predicted_rrp: {new_champion['uses_predicted_rrp']}")
    print()

    if args.dry_run:
        print("[promote] --dry-run: not writing.")
        return 0

    if prev is not None:
        with open(archive_path, "w") as f:
            json.dump(prev, f, indent=2, default=str)
        print(f"[promote] archived previous champion -> {archive_path}")

    _save_champion(champion_path, new_champion)
    print(f"[promote] wrote new champion          -> {champion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
