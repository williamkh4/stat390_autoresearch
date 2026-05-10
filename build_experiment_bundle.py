"""
Build the five-deliverable analysis bundle for any controlled-experiment set.

Reads a results CSV in the schema produced by `run_controlled_experiments.py`
or `run_stress_test.py` (one row per experiment, including
val + test mse/rmse/mae and a `series` column), and writes:

    1. INDEX.md             -- bundle table of contents (the "experiment log")
    2. metric_trajectory.png -- val + test RMSE / MAE per experiment, grouped
                                by series, with baseline reference lines
    3. keep_discard_crash.md -- Keep / Discard / Crash categorisation per row
    4. best_vs_baseline.md   -- best experiment side-by-side with baselines
    5. what_worked_memo.md   -- data-driven narrative of which knobs moved
                                the needle, written for a research reader

The bundle is *self-contained*: pass any controlled-style CSV (with a
`baseline` series row for each seasonal naive) and you get a folder you
can hand to a reader without further explanation.

Usage:
    python build_experiment_bundle.py \
        --results-csv experiments/controlled/controlled_results.csv \
        --out-dir analysis/controlled_bundle \
        --title "Controlled Experiments (Series A/B/C)"
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- helpers ---------------------------------------------------------

# Treat a candidate as "Keep" if its validation MSE is below the strong
# (yearly) baseline by at least this fraction. Anything weaker we call
# Discard. Errors are Crash. The threshold is small but non-zero so that
# results that only nudge below the baseline don't get oversold.
KEEP_MARGIN_FRAC = 0.01

STRONG_BASELINE_NAME = "seasonal_naive_364"
WEAK_BASELINE_NAME = "seasonal_naive_7"


def _fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.0f}"


def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.1f}%"


def _split_baselines(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_mask = df["series"].astype(str).str.lower() == "baseline"
    return df[base_mask].copy(), df[~base_mask].copy()


def _baseline_metrics(baselines: pd.DataFrame, name: str) -> Dict[str, float]:
    """Return val/test mse/rmse/mae for the named baseline (or empty dict)."""
    row = baselines[baselines["label"] == name]
    if not len(row):
        return {}
    r = row.iloc[0]
    return {
        "val_mse": float(r.get("mse_demand")) if pd.notna(r.get("mse_demand")) else None,
        "val_rmse": float(r.get("rmse_demand")) if pd.notna(r.get("rmse_demand")) else None,
        "val_mae": float(r.get("mae_demand")) if pd.notna(r.get("mae_demand")) else None,
        "test_mse": float(r.get("mse_demand_test")) if pd.notna(r.get("mse_demand_test")) else None,
        "test_rmse": float(r.get("rmse_demand_test")) if pd.notna(r.get("rmse_demand_test")) else None,
        "test_mae": float(r.get("mae_demand_test")) if pd.notna(r.get("mae_demand_test")) else None,
    }


# ---------- 1. INDEX.md (bundle log) ----------------------------------------

def write_index(out_dir: Path, title: str, source_csv: Path,
                df: pd.DataFrame, baselines: pd.DataFrame) -> Path:
    base, body = _split_baselines(df)
    series_counts = body["series"].value_counts().sort_index()
    crashes = int((body["error"].fillna("") != "").sum())

    lines: List[str] = []
    lines.append(f"# {title} — Experiment Log Bundle")
    lines.append("")
    lines.append(
        "Self-contained analysis package for one controlled experiment set. "
        "Read this file first; the per-deliverable files below are the "
        "answers to specific questions about the run."
    )
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append(f"- Results CSV: `{source_csv}`")
    lines.append(f"- Total experiments: **{len(body)}**")
    lines.append(f"- Series: **{len(series_counts)}** "
                 f"(" + ", ".join(f"{s}={n}" for s, n in series_counts.items()) + ")")
    lines.append(f"- Crashes: **{crashes}**")
    if len(base):
        lines.append(f"- Baselines included: " +
                     ", ".join(f"`{n}`" for n in base["label"].tolist()))
    lines.append("")
    lines.append("## Deliverables")
    lines.append("")
    lines.append("| # | File | What it answers |")
    lines.append("|---|---|---|")
    lines.append("| 1 | `INDEX.md` (this file) | What's in this bundle, where each artifact lives |")
    lines.append("| 2 | `metric_trajectory.png` | How val + test RMSE/MAE move across the sweep, vs. baselines |")
    lines.append("| 3 | `keep_discard_crash.md` | For every experiment: keep / discard / crash, with reason |")
    lines.append("| 4 | `best_vs_baseline.md` | The single best experiment side-by-side with the baselines |")
    lines.append("| 5 | `what_worked_memo.md` | Per-series narrative: which knob moved the needle, and which didn't |")
    lines.append("")
    lines.append("## Per-experiment summary")
    lines.append("")
    cols = ["series", "label", "model_type", "n_features",
            "mse_demand", "rmse_demand", "mae_demand",
            "mse_demand_test", "rmse_demand_test", "mae_demand_test",
            "runtime_sec", "error"]
    avail = [c for c in cols if c in df.columns]
    lines.append("| " + " | ".join(avail) + " |")
    lines.append("|" + "|".join(["---"] * len(avail)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in avail:
            v = r.get(c)
            if pd.isna(v):
                cells.append("—")
            elif isinstance(v, float):
                if c.endswith("_test") or c in ("mse_demand", "rmse_demand", "mae_demand"):
                    cells.append(f"{v:,.0f}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    p = out_dir / "INDEX.md"
    p.write_text("\n".join(lines))
    return p


# ---------- 2. metric_trajectory.png ----------------------------------------

def write_trajectory(out_dir: Path, title: str,
                     body: pd.DataFrame, baselines: pd.DataFrame) -> Path:
    """Two-panel plot: RMSE and MAE per experiment, val + test on each panel.

    Vertical separators delineate series. Horizontal reference lines show
    the strong baseline's val + test metrics so the reader can see at a
    glance how far each experiment is from the floor.
    """
    body = body.reset_index(drop=True).copy()
    body["x"] = np.arange(len(body))

    # Mark series boundaries (where the series label changes between rows)
    series_boundaries = []
    series_centers = {}
    prev = None
    series_to_xs: Dict[str, List[int]] = {}
    for i, s in enumerate(body["series"].astype(str).tolist()):
        series_to_xs.setdefault(s, []).append(i)
        if prev is not None and s != prev:
            series_boundaries.append(i - 0.5)
        prev = s
    for s, xs in series_to_xs.items():
        series_centers[s] = (xs[0] + xs[-1]) / 2

    # Reference baseline metrics
    strong = _baseline_metrics(baselines, STRONG_BASELINE_NAME)

    # Normalise the error column so masks work whether the input had NaN
    # cells or stringified "nan" cells.
    body["error"] = body["error"].apply(_err_str)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.55 * len(body) + 4), 9.5),
                             sharex=True)

    for ax, metric, suffix, ylabel in [
        (axes[0], "rmse_demand", "_test", "RMSE of demand (log scale)"),
        (axes[1], "mae_demand", "_test", "MAE of demand (log scale)"),
    ]:
        val_col = metric
        test_col = metric + suffix

        # Crashes show as a hollow red marker on the val line.
        crashed_mask = body["error"] != ""
        ok_mask = ~crashed_mask

        ax.plot(body.loc[ok_mask, "x"], body.loc[ok_mask, val_col],
                color="#1f77b4", marker="o", markersize=6, linewidth=1.5,
                label="validation", zorder=3)
        ax.plot(body.loc[ok_mask, "x"], body.loc[ok_mask, test_col],
                color="#d62728", marker="s", markersize=6, linewidth=1.5,
                linestyle="--", label="test", zorder=3)

        # Mark crashes (vertical span)
        for xc in body.loc[crashed_mask, "x"].tolist():
            ax.axvspan(xc - 0.4, xc + 0.4, color="black", alpha=0.08, zorder=1)
            ax.scatter([xc], [body[val_col].dropna().median()], marker="x",
                       color="black", s=80, label="crash", zorder=4)

        # Baseline reference (strong baseline)
        if strong:
            if metric == "rmse_demand":
                v_ref, t_ref = strong["val_rmse"], strong["test_rmse"]
            else:
                v_ref, t_ref = strong["val_mae"], strong["test_mae"]
            if v_ref is not None:
                ax.axhline(v_ref, color="#1f77b4", linestyle=":", linewidth=1,
                           alpha=0.7,
                           label=f"baseline val ({v_ref:,.0f})")
            if t_ref is not None:
                ax.axhline(t_ref, color="#d62728", linestyle=":", linewidth=1,
                           alpha=0.7,
                           label=f"baseline test ({t_ref:,.0f})")

        # Series boundaries
        for xb in series_boundaries:
            ax.axvline(xb, color="gray", linestyle="-", linewidth=0.7, alpha=0.4)

        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        # Legend off to the right so it never sits on top of the series labels.
        ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
                  fontsize=8, framealpha=0.9)

    # Series labels: place them directly above the upper panel using a
    # blended transform (data-x, axes-y), so log-scale y doesn't move them.
    blended = axes[0].get_xaxis_transform()
    for s, cx in series_centers.items():
        axes[0].text(cx, 1.04, f"series {s}", ha="center", va="bottom",
                     transform=blended, fontsize=10, color="#222", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.25",
                               facecolor="white", edgecolor="#888", alpha=0.95),
                     clip_on=False)

    # X-axis tick labels = experiment labels
    axes[1].set_xticks(body["x"])
    axes[1].set_xticklabels(body["label"].astype(str), rotation=45, ha="right",
                            fontsize=8)
    axes[1].set_xlabel("experiment")

    fig.suptitle(f"{title}: validation + test metric trajectory", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))

    p = out_dir / "metric_trajectory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------- 3. keep_discard_crash.md ----------------------------------------

def _err_str(v) -> str:
    """Normalise the error column: NaN / None / 'nan' / '' all map to ''."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def categorise(row: pd.Series, strong_val_mse: Optional[float]) -> Tuple[str, str]:
    """Return (category, reason) for a single experiment row."""
    err = _err_str(row.get("error"))
    if err:
        return "Crash", f"errored during fit/predict: `{err}`"
    val_mse = row.get("mse_demand")
    if val_mse is None or (isinstance(val_mse, float) and np.isnan(val_mse)):
        return "Crash", "no validation MSE recorded"
    if strong_val_mse is None:
        return "Keep", "no strong baseline available; defaulting to Keep"
    margin = (strong_val_mse - val_mse) / strong_val_mse
    if margin >= KEEP_MARGIN_FRAC:
        return "Keep", (f"val MSE {val_mse:,.0f} beats strong baseline "
                        f"{strong_val_mse:,.0f} by {margin*100:.1f}%")
    return "Discard", (f"val MSE {val_mse:,.0f} does not beat strong baseline "
                       f"{strong_val_mse:,.0f} (margin {margin*100:+.1f}%)")


def write_keep_discard_crash(out_dir: Path, title: str,
                             body: pd.DataFrame,
                             baselines: pd.DataFrame) -> Path:
    strong = _baseline_metrics(baselines, STRONG_BASELINE_NAME)
    strong_val_mse = strong.get("val_mse")

    cats: List[str] = []
    reasons: List[str] = []
    for _, r in body.iterrows():
        c, why = categorise(r, strong_val_mse)
        cats.append(c)
        reasons.append(why)
    body = body.copy()
    body["category"] = cats
    body["reason"] = reasons

    n_keep = int((body["category"] == "Keep").sum())
    n_disc = int((body["category"] == "Discard").sum())
    n_crash = int((body["category"] == "Crash").sum())

    lines: List[str] = []
    lines.append(f"# {title} — Keep / Discard / Crash")
    lines.append("")
    lines.append("Each experiment is sorted into one of three buckets:")
    lines.append("")
    lines.append("- **Keep** — validation MSE beat the strong baseline "
                 f"(`{STRONG_BASELINE_NAME}`) by at least "
                 f"{KEEP_MARGIN_FRAC*100:.0f}%. Worth carrying into the next "
                 "iteration / reporting in a paper.")
    lines.append("- **Discard** — fit ran fine but did not beat the strong "
                 "baseline by a meaningful margin. The configuration is logged "
                 "but should not be promoted.")
    lines.append("- **Crash** — fit/predict raised; the row's error message is the "
                 "evidence trail.")
    lines.append("")
    if strong_val_mse is not None:
        lines.append(f"Strong baseline (`{STRONG_BASELINE_NAME}`) val MSE: "
                     f"**{strong_val_mse:,.0f}**.")
    lines.append("")
    lines.append(f"Counts: **Keep {n_keep}** | **Discard {n_disc}** | **Crash {n_crash}**.")
    lines.append("")

    for cat in ("Keep", "Discard", "Crash"):
        sub = body[body["category"] == cat]
        if not len(sub):
            continue
        lines.append(f"## {cat} ({len(sub)})")
        lines.append("")
        lines.append("| series | label | model | val MSE | test MSE | reason |")
        lines.append("|---|---|---|---:|---:|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['series']} "
                f"| `{r['label']}` "
                f"| {r.get('model_type', '')} "
                f"| {_fmt_int(r.get('mse_demand'))} "
                f"| {_fmt_int(r.get('mse_demand_test'))} "
                f"| {r['reason']} |"
            )
        lines.append("")

    p = out_dir / "keep_discard_crash.md"
    p.write_text("\n".join(lines))
    return p


# ---------- 4. best_vs_baseline.md ------------------------------------------

def write_best_vs_baseline(out_dir: Path, title: str,
                           body: pd.DataFrame,
                           baselines: pd.DataFrame) -> Path:
    ok = body[body["error"].fillna("") == ""].copy()
    if not len(ok):
        p = out_dir / "best_vs_baseline.md"
        p.write_text(f"# {title} — Best vs. Baseline\n\n"
                     "No successful experiments to compare.\n")
        return p

    best = ok.sort_values("mse_demand").iloc[0]
    strong = _baseline_metrics(baselines, STRONG_BASELINE_NAME)
    weak = _baseline_metrics(baselines, WEAK_BASELINE_NAME)

    def _delta_pct(best_v, base_v):
        if best_v is None or base_v is None or base_v == 0:
            return None
        return 100.0 * (base_v - best_v) / base_v

    rows: List[Dict[str, str]] = []
    metric_pairs = [
        ("MSE",  "mse_demand",   "mse_demand_test",   "val_mse",  "test_mse"),
        ("RMSE", "rmse_demand",  "rmse_demand_test",  "val_rmse", "test_rmse"),
        ("MAE",  "mae_demand",   "mae_demand_test",   "val_mae",  "test_mae"),
    ]

    lines: List[str] = []
    lines.append(f"# {title} — Best Result vs. Baseline")
    lines.append("")
    lines.append(
        f"The single best experiment in this bundle is **`{best['label']}`** "
        f"(series {best['series']}, model `{best.get('model_type', '?')}`)."
    )
    lines.append("")
    if "feature_config" in best.index:
        lines.append(f"- features: `{best['feature_config']}`")
    if "n_features" in best.index and pd.notna(best.get("n_features")):
        lines.append(f"- n_features: {int(best['n_features'])}")
    if "runtime_sec" in best.index and pd.notna(best.get("runtime_sec")):
        lines.append(f"- runtime: {float(best['runtime_sec']):.2f}s")
    lines.append("")
    lines.append("## Side-by-side")
    lines.append("")
    lines.append(
        "| metric | best (val) | best (test) | "
        f"`{WEAK_BASELINE_NAME}` (val) | `{WEAK_BASELINE_NAME}` (test) | "
        f"`{STRONG_BASELINE_NAME}` (val) | `{STRONG_BASELINE_NAME}` (test) | "
        "Δ vs. strong (val) | Δ vs. strong (test) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, val_col, test_col, strong_val_key, strong_test_key in metric_pairs:
        bv = best.get(val_col)
        bt = best.get(test_col)
        wv = weak.get(strong_val_key) if weak else None
        wt = weak.get(strong_test_key) if weak else None
        sv = strong.get(strong_val_key) if strong else None
        st = strong.get(strong_test_key) if strong else None
        d_val = _delta_pct(bv, sv)
        d_test = _delta_pct(bt, st)
        lines.append(
            f"| {label} "
            f"| {_fmt_int(bv)} | {_fmt_int(bt)} "
            f"| {_fmt_int(wv)} | {_fmt_int(wt)} "
            f"| {_fmt_int(sv)} | {_fmt_int(st)} "
            f"| {_fmt_pct(d_val)} | {_fmt_pct(d_test)} |"
        )
    lines.append("")
    lines.append("Δ is `(baseline − best) / baseline`, so positive = best is "
                 "better than baseline.")
    lines.append("")

    # Top-5 leaderboard for context
    lines.append("## Top 5 by validation MSE")
    lines.append("")
    lines.append("| rank | series | label | model | val MSE | test MSE | runtime |")
    lines.append("|---:|---|---|---|---:|---:|---:|")
    top5 = ok.sort_values("mse_demand").head(5)
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        lines.append(
            f"| {i} | {r['series']} | `{r['label']}` | {r.get('model_type', '')} "
            f"| {_fmt_int(r.get('mse_demand'))} | {_fmt_int(r.get('mse_demand_test'))} "
            f"| {float(r.get('runtime_sec', 0.0)):.2f}s |"
        )
    lines.append("")

    p = out_dir / "best_vs_baseline.md"
    p.write_text("\n".join(lines))
    return p


# ---------- 5. what_worked_memo.md ------------------------------------------

def _series_summary(series_label: str, sub: pd.DataFrame) -> Dict[str, object]:
    """Compute best/worst/spread for a single series (errors removed)."""
    ok = sub[sub["error"].fillna("") == ""].copy()
    out: Dict[str, object] = {
        "series": series_label,
        "n": len(sub),
        "n_ok": len(ok),
        "n_crash": int((sub["error"].fillna("") != "").sum()),
    }
    if not len(ok):
        return out
    best = ok.sort_values("mse_demand").iloc[0]
    worst = ok.sort_values("mse_demand").iloc[-1]
    out.update({
        "best_label": best["label"],
        "best_val_mse": float(best["mse_demand"]),
        "best_test_mse": float(best.get("mse_demand_test")) if pd.notna(best.get("mse_demand_test")) else None,
        "worst_label": worst["label"],
        "worst_val_mse": float(worst["mse_demand"]),
        "spread_pct": (100.0 * (worst["mse_demand"] - best["mse_demand"]) / worst["mse_demand"])
                      if worst["mse_demand"] else None,
    })
    return out


def write_what_worked_memo(out_dir: Path, title: str,
                           body: pd.DataFrame,
                           baselines: pd.DataFrame) -> Path:
    strong = _baseline_metrics(baselines, STRONG_BASELINE_NAME)
    strong_val_mse = strong.get("val_mse")

    series_summaries: List[Dict[str, object]] = []
    for s, sub in body.groupby("series"):
        series_summaries.append(_series_summary(str(s), sub))
    series_summaries.sort(key=lambda x: str(x["series"]))

    ok = body[body["error"].fillna("") == ""].copy()
    overall_best = ok.sort_values("mse_demand").iloc[0] if len(ok) else None
    n_keep = 0
    if strong_val_mse is not None:
        n_keep = int((ok["mse_demand"] < strong_val_mse * (1 - KEEP_MARGIN_FRAC)).sum())

    lines: List[str] = []
    lines.append(f"# {title} — What Actually Worked")
    lines.append("")
    lines.append("A 1-page memo distilled from the per-experiment table. The "
                 "structure mirrors the sweep design: each series is one knob, "
                 "so the question for each is *did moving this knob help, and "
                 "by how much?*")
    lines.append("")

    # Headline
    if overall_best is not None:
        lines.append("## Headline")
        lines.append("")
        lines.append(
            f"- Best of bundle: **`{overall_best['label']}`** "
            f"(series {overall_best['series']}, model "
            f"`{overall_best.get('model_type', '?')}`).")
        lines.append(
            f"  - validation MSE: **{_fmt_int(overall_best['mse_demand'])}** "
            f"(test MSE {_fmt_int(overall_best.get('mse_demand_test'))})")
        if strong_val_mse is not None:
            d = 100 * (strong_val_mse - overall_best["mse_demand"]) / strong_val_mse
            lines.append(
                f"  - vs. `{STRONG_BASELINE_NAME}` ({_fmt_int(strong_val_mse)} on val): "
                f"**{_fmt_pct(d)}** improvement.")
        lines.append(f"- Of {len(ok)} successful experiments, **{n_keep}** beat "
                     f"the strong baseline by ≥{KEEP_MARGIN_FRAC*100:.0f}%.")
        lines.append("")

    # Per-framework summary, only when more than one model framework is present.
    frameworks = sorted(set(ok["model_type"].astype(str).tolist()))
    frameworks = [f for f in frameworks if f and f != "baseline"]
    if len(frameworks) > 1:
        lines.append("## Per-framework summary")
        lines.append("")
        lines.append("| framework | best label | val MSE | test MSE | "
                     "median val MSE | n experiments |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for fw in frameworks:
            sub = ok[ok["model_type"] == fw]
            if not len(sub):
                continue
            b = sub.sort_values("mse_demand").iloc[0]
            med_val = float(sub["mse_demand"].median())
            lines.append(
                f"| `{fw}` "
                f"| `{b['label']}` "
                f"| {_fmt_int(b.get('mse_demand'))} "
                f"| {_fmt_int(b.get('mse_demand_test'))} "
                f"| {_fmt_int(med_val)} "
                f"| {len(sub)} |"
            )
        lines.append("")

    # Per-series narrative
    lines.append("## Per-series read-out")
    lines.append("")
    for s in series_summaries:
        lines.append(f"### Series {s['series']} ({s['n']} experiments, "
                     f"{s.get('n_ok', 0)} successful, {s.get('n_crash', 0)} crashes)")
        lines.append("")
        if "best_label" not in s:
            lines.append("All experiments crashed; nothing to compare. See "
                         "`keep_discard_crash.md` for the error trail.")
            lines.append("")
            continue

        lines.append(
            f"- best: `{s['best_label']}` (val MSE {_fmt_int(s['best_val_mse'])}"
            + (f", test MSE {_fmt_int(s['best_test_mse'])}" if s.get("best_test_mse") is not None else "")
            + ")"
        )
        lines.append(
            f"- worst: `{s['worst_label']}` (val MSE {_fmt_int(s['worst_val_mse'])})"
        )
        if s.get("spread_pct") is not None:
            lines.append(
                f"- spread within series: **{s['spread_pct']:.1f}%** of the worst "
                f"value — the bigger this number, the more sensitive the metric "
                f"is to *this* knob."
            )
        # Also list each row in the series so the reader can scan
        sub = body[body["series"] == s["series"]].copy()
        if len(sub):
            lines.append("")
            lines.append("| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                lines.append(
                    f"| `{r['label']}` "
                    f"| {_fmt_int(r.get('mse_demand'))} "
                    f"| {_fmt_int(r.get('mse_demand_test'))} "
                    f"| {_fmt_int(r.get('rmse_demand'))} "
                    f"| {_fmt_int(r.get('rmse_demand_test'))} "
                    f"| {_fmt_int(r.get('mae_demand'))} "
                    f"| {_fmt_int(r.get('mae_demand_test'))} |"
                )
            lines.append("")

    # Reading guide
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- Within a series, **everything except one knob is held fixed**, so a "
        "spread tells you how sensitive the metric is to that knob alone."
    )
    lines.append(
        "- The validation column is the optimisation surface; the test column "
        "is reported for honesty. A series where val improves but test doesn't "
        "(or moves the wrong way) is a generalisation warning, not a win."
    )
    lines.append(
        "- Beating the strong baseline (`seasonal_naive_364`, yearly recall) is "
        "the floor for *adding research value*; anything weaker means weather + "
        "lag features didn't pay off for that configuration."
    )
    lines.append("")

    # Caveats specifically about the test number
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The framework is deterministic, so a single number per experiment is "
        "a point estimate without a noise floor. Treat differences smaller than "
        "~5% with skepticism. (See `FAILURE_ANALYSIS_MEMO.docx` for the recommendation "
        "to add a bootstrap noise estimate before reading any single number as definitive.)"
    )
    lines.append(
        "- Test scores here come from a **train-only** fit predicting on the "
        "locked test window — same protocol as the auto loop's "
        "`--evaluate-on-test`. The deployment-style refit on train+val is in "
        "`run_test_evaluation.py`."
    )
    lines.append(
        "- The auto loop's champion promotion is decided on validation MSE only; "
        "test numbers in this bundle are diagnostic, not selectorial."
    )
    lines.append("")

    p = out_dir / "what_worked_memo.md"
    p.write_text("\n".join(lines))
    return p


# ---------- driver ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build experiment bundle.")
    parser.add_argument("--results-csv", required=True,
                        help="Path to a controlled-style results CSV.")
    parser.add_argument("--out-dir", required=True,
                        help="Folder to write the 5 deliverables into.")
    parser.add_argument("--title", default="Experiment Bundle",
                        help="Human-readable title prefix used in the deliverables.")
    args = parser.parse_args()

    src = Path(args.results_csv)
    if not src.exists():
        raise SystemExit(f"results CSV not found: {src}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    # Backwards compat: ensure required columns exist
    for col in ("series", "label", "error", "model_type"):
        if col not in df.columns:
            df[col] = ""
    for col in ("mse_demand", "rmse_demand", "mae_demand",
                "mse_demand_test", "rmse_demand_test", "mae_demand_test",
                "n_features", "runtime_sec"):
        if col not in df.columns:
            df[col] = np.nan

    # Normalise the error column so every place that does `.fillna("") != ""`
    # works whether pandas read empty cells as NaN floats or as empty strings.
    df["error"] = df["error"].apply(_err_str)

    baselines, body = _split_baselines(df)

    wrote: List[Path] = []
    wrote.append(write_index(out_dir, args.title, src, df, baselines))
    wrote.append(write_trajectory(out_dir, args.title, body, baselines))
    wrote.append(write_keep_discard_crash(out_dir, args.title, body, baselines))
    wrote.append(write_best_vs_baseline(out_dir, args.title, body, baselines))
    wrote.append(write_what_worked_memo(out_dir, args.title, body, baselines))

    print(f"[bundle] wrote {len(wrote)} files to {out_dir}/")
    for p in wrote:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
