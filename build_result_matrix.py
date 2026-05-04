"""
Build the experiment-result matrix from the AutoResearch master log.

Pivots `experiments/auto_runs/master_log.csv` into a (feature_preset x
model_type) table of best validation MSE per pair. Writes both a CSV
(machine-readable) and a Markdown table (human-readable) to
`analysis/`. Re-run after each auto-iteration to keep the matrix
current.

Usage:
    python build_result_matrix.py
    python build_result_matrix.py --results-dir other/runs --out analysis/
"""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build result matrix.")
    parser.add_argument("--results-dir", default="experiments/auto_runs",
                        help="Folder containing master_log.csv.")
    parser.add_argument("--out", default="analysis",
                        help="Output folder for result_matrix.{csv,md}.")
    args = parser.parse_args()

    log_path = Path(args.results_dir) / "master_log.csv"
    if not log_path.exists():
        raise SystemExit(f"No master log at {log_path}.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    df = df[df["error"].fillna("") == ""].copy()
    df["model_type"] = df["candidate_name"].str.split("__").str[0]
    df["feature_preset"] = df["candidate_name"].str.split("__").str[1]

    base_mask = df["candidate_name"].isin(["seasonal_naive_7", "seasonal_naive_364"])
    baselines = df[base_mask].groupby("candidate_name")["mse_demand"].first().to_dict()
    body = df[~base_mask].copy()

    best = (body.groupby(["feature_preset", "model_type"])["mse_demand"]
                 .min().reset_index())
    matrix = best.pivot(index="feature_preset", columns="model_type",
                        values="mse_demand")

    # Sort rows by their min across model types (best preset first).
    matrix["__best"] = matrix.min(axis=1)
    matrix = matrix.sort_values("__best").drop(columns="__best")

    # Reorder columns: linear -> rf -> mlp -> gbm
    preferred = ["numpy_ols", "ridge", "rf", "mlp", "gbm"]
    matrix = matrix[[c for c in preferred if c in matrix.columns]]

    csv_path = out_dir / "result_matrix.csv"
    matrix.to_csv(csv_path)

    def fmt(v):
        return "—" if pd.isna(v) else f"{v/1e6:.2f}M"

    overall_min = matrix.min().min()
    md_path = out_dir / "result_matrix.md"
    with open(md_path, "w") as f:
        f.write("# Experiment-Result Matrix\n\n")
        f.write(
            "Best validation MSE achieved for each (feature_preset, "
            "model_type) combination across all AutoResearch runs in "
            "`experiments/auto_runs/master_log.csv`. Cells are the "
            "**minimum** MSE seen for that pair across all hyperparameter "
            "variants tried; `—` means the combination has not been "
            "explored yet.\n\n"
        )
        f.write(f"- Runs aggregated: **{df['run_id'].nunique()}**\n")
        f.write(f"- Total non-baseline candidates: **{len(body)}** "
                f"({body['candidate_name'].nunique()} unique)\n")
        if baselines:
            f.write(
                f"- Reference baselines: "
                f"`seasonal_naive_7` = "
                f"{baselines.get('seasonal_naive_7', float('nan'))/1e6:.1f}M, "
                f"`seasonal_naive_364` = "
                f"{baselines.get('seasonal_naive_364', float('nan'))/1e6:.1f}M\n"
            )
        f.write("\n")
        f.write("| feature_preset | " + " | ".join(matrix.columns) + " |\n")
        f.write("|---" + ("|---:" * len(matrix.columns)) + "|\n")
        for fp, row in matrix.iterrows():
            cells = []
            for col in matrix.columns:
                v = row[col]
                cell = fmt(v)
                if pd.notna(v) and abs(v - overall_min) < 1.0:
                    cell = f"**{cell}**"
                cells.append(cell)
            f.write(f"| `{fp}` | " + " | ".join(cells) + " |\n")
        f.write("\n**Bold** cell = current overall champion.\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Matrix shape: {matrix.shape[0]} feature_presets x "
          f"{matrix.shape[1]} model_types")


if __name__ == "__main__":
    main()
