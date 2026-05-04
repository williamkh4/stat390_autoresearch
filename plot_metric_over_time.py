"""
Plot how validation MSE evolved across AutoResearch runs.

Reads `experiments/auto_runs/master_log.csv` and produces a single PNG
that shows three series across run number (chronological):

  * "champion-after-run" -- the running minimum MSE; this is the
    "official" leaderboard line, since the champion only moves down
  * "best-of-run"        -- the best MSE produced inside that single
    run (regardless of whether it beat the prior champion)
  * "all candidates"     -- a scatter of every candidate's MSE,
    color-coded by model_type (gbm/rf/mlp/ridge/numpy_ols/baselines)

Two reference horizontal lines mark the seasonal-naive baselines so the
"is anything beating yearly recall yet?" question is a glance.

The y-axis is log-scaled because MSE values span almost 1.5 orders of
magnitude across the candidate pool.

Output:
    analysis/metric_over_time.png

Usage:
    python plot_metric_over_time.py
    python plot_metric_over_time.py --results-dir my/results --out my/plot.png
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MSE over runs.")
    parser.add_argument("--results-dir", default="experiments/auto_runs",
                        help="Folder containing master_log.csv.")
    parser.add_argument("--out", default="analysis/metric_over_time.png",
                        help="Output PNG path.")
    args = parser.parse_args()

    log_path = Path(args.results_dir) / "master_log.csv"
    if not log_path.exists():
        raise SystemExit(f"No master log at {log_path}.")

    df = pd.read_csv(log_path)
    df = df[df["error"].fillna("") == ""].copy()
    df["model_type"] = df["candidate_name"].str.split("__").str[0]

    # Order runs chronologically and assign an integer index for plotting.
    run_order = (
        df.groupby("run_id")["timestamp_utc"].min()
          .sort_values().index.tolist()
    )
    df["run_idx"] = df["run_id"].map({rid: i + 1 for i, rid in enumerate(run_order)})

    # Per-run aggregates
    per_run = (
        df.groupby("run_idx")
          .agg(best_of_run=("mse_demand", "min"),
               run_id=("run_id", "first"))
          .reset_index()
          .sort_values("run_idx")
    )
    per_run["champion_after_run"] = per_run["best_of_run"].cummin()

    # Reference baselines (last value in master log; they're constant)
    base_mask = df["candidate_name"].isin(["seasonal_naive_7", "seasonal_naive_364"])
    baselines = df[base_mask].groupby("candidate_name")["mse_demand"].first().to_dict()

    # ----- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    # Scatter of every non-baseline candidate, color-coded by model_type.
    # Baselines are drawn as horizontal reference lines instead.
    color_map = {
        "gbm":       "#d62728",
        "mlp":       "#1f77b4",
        "rf":        "#2ca02c",
        "ridge":     "#ff7f0e",
        "numpy_ols": "#9467bd",
    }
    cand = df[~base_mask].copy()
    for mt, sub in cand.groupby("model_type"):
        ax.scatter(sub["run_idx"], sub["mse_demand"],
                   s=20, alpha=0.45, label=mt,
                   color=color_map.get(mt, "#888"))

    # "Best of each run" (orange line) and running champion (black line).
    ax.plot(per_run["run_idx"], per_run["best_of_run"],
            color="orange", linewidth=1.4, marker="o", markersize=5,
            label="best of run", zorder=3)
    ax.plot(per_run["run_idx"], per_run["champion_after_run"],
            color="black", linewidth=2.2, marker="s", markersize=6,
            label="champion after run", zorder=4)

    # Reference baselines.
    if "seasonal_naive_364" in baselines:
        ax.axhline(baselines["seasonal_naive_364"], color="gray",
                   linestyle="--", linewidth=1,
                   label=f"seasonal_naive_364 ({baselines['seasonal_naive_364']/1e6:.1f}M)")
    if "seasonal_naive_7" in baselines:
        ax.axhline(baselines["seasonal_naive_7"], color="lightgray",
                   linestyle=":", linewidth=1,
                   label=f"seasonal_naive_7 ({baselines['seasonal_naive_7']/1e6:.0f}M)")

    ax.set_yscale("log")
    ax.set_xlabel("AutoResearch run (chronological order)")
    ax.set_ylabel("validation MSE of demand  (log scale)")
    n_runs = len(per_run)
    n_cands = len(cand)
    ax.set_title(
        f"Validation MSE across AutoResearch runs ({n_runs} runs, {n_cands} model fits)"
    )
    ax.set_xticks(per_run["run_idx"])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote: {out_path}")
    print(f"  runs plotted: {n_runs}")
    print(f"  candidates plotted: {n_cands}")
    print(f"  champion after final run: {per_run['champion_after_run'].iloc[-1]:,.0f}")


if __name__ == "__main__":
    main()
