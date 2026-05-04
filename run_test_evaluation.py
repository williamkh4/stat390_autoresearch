"""
One-shot evaluation of the current champion on the locked 365-day test set.

The AutoResearch loop and `run_baseline.py` only ever read `Splits.train`
and `Splits.val`. The test split is constructed by `make_splits()` and
then deliberately ignored everywhere else, so model selection cannot
overfit to it. This script is the *only* place that breaks that wall,
and it does so on purpose: re-fit the champion on (train + val) and
report a single number for how it does on the held-out test window.

Why a separate entry point: every time the test set is used for
evaluation, its credibility as a held-out estimator goes down a little
(see `ERROR_TAXONOMY.md` item L5). The script therefore:

  * Refuses to run multiple times on the same champion by default.
    Override with `--force`.
  * Logs every evaluation to `experiments/test_evaluation/test_log.csv`
    so the project can later report "we spent the test set N times."
  * Also evaluates both seasonal-naive baselines on test, refit on
    (train + val), so the champion's test number has a contemporaneous
    reference (not just the val baseline number we already know).

Outputs:
    experiments/test_evaluation/<run_id>__<timestamp>.json   full report
    experiments/test_evaluation/test_log.csv                 append-only log

Usage:
    python run_test_evaluation.py
    python run_test_evaluation.py --force          # allow repeat eval
    python run_test_evaluation.py --champion-path other/champion.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.autoresearch import (
    Candidate,
    CHAMPION_NAME,
    MASTER_LOG_NAME,
    _full_search_space,
    _load_champion,
    _make_factory,
    baseline_candidates,
)
from src.baselines import SeasonalNaive
from src.data_loader import load_merged
from src.features import FeatureConfig, build_features
from src.metrics import PRIMARY_METRIC_NAME, score_all
from src.split import Splits, make_splits


TEST_LOG_NAME = "test_log.csv"
TEST_DIR_DEFAULT = Path("experiments/test_evaluation")


# ----------------------------------------------------------------------------
# Reconstructing the champion candidate from champion.json
# ----------------------------------------------------------------------------

def _candidate_from_champion(champion: dict) -> Candidate:
    """Find the candidate registered under the champion's name.

    Names are deterministic across the search space, so the lookup is
    exact for any champion the auto-generator produced. If the champion
    was set under a legacy name (pre-auto-naming), we fall back to
    matching by feature_config signature -- with a clear log line when
    that happens, so the substitution is auditable.
    """
    name = champion["name"]
    space = _full_search_space()
    by_name = {c.name: c for c in space}
    if name in by_name:
        return by_name[name]

    # Legacy fallback: match by feature_config signature.
    fc_dict = champion.get("feature_config") or {}
    target = (
        bool(fc_dict.get("use_calendar", False)),
        bool(fc_dict.get("use_temp", False)),
        bool(fc_dict.get("use_apparent_temp", False)),
        bool(fc_dict.get("use_rrp", False)),
        tuple(fc_dict.get("demand_lags", []) or []),
        tuple(fc_dict.get("rolling_windows", []) or []),
    )
    for c in space:
        sig = (
            c.feature_config.use_calendar, c.feature_config.use_temp,
            c.feature_config.use_apparent_temp, c.feature_config.use_rrp,
            tuple(c.feature_config.demand_lags),
            tuple(c.feature_config.rolling_windows),
        )
        if sig == target:
            print(f"[test-eval] WARNING: champion name '{name}' not in current "
                  f"search space; remapped by feature signature to '{c.name}'.")
            return c
    raise SystemExit(
        f"Could not reconstruct candidate for champion '{name}'. "
        "Has MODEL_SPECS or FEATURE_PRESETS changed since the champion was set? "
        "Re-run AutoResearch to elect a new champion under the current space, "
        "then re-run this script."
    )


# ----------------------------------------------------------------------------
# Test-set scoring (the only place that touches Splits.test)
# ----------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    metrics: dict
    runtime_sec: float
    n_features: int
    n_fit: int          # rows used to fit (train + val)
    n_test: int         # rows predicted (test)
    error: str | None = None


def _score_on_test(candidate: Candidate, splits: Splits) -> TestResult:
    """Re-fit `candidate` on train+val, predict on test, return metrics.

    Mirrors the logic of `autoresearch.score_candidate` but with the
    boundary moved one window forward: fit on train+val, predict on
    the locked test rows.
    """
    t0 = time.perf_counter()
    try:
        model = candidate.model_factory()

        if isinstance(model, SeasonalNaive):
            history = pd.concat([splits.train, splits.val]).reset_index(drop=True)
            model.fit(history["time"], history["demand"].to_numpy())
            y_pred = model.predict(splits.test["time"])
            y_true = splits.test["demand"].to_numpy(dtype=float)
            metrics = score_all(y_true, y_pred)
            return TestResult(
                name=candidate.name, metrics=metrics,
                runtime_sec=time.perf_counter() - t0,
                n_features=0, n_fit=len(history), n_test=len(splits.test),
            )

        fc: FeatureConfig = candidate.feature_config
        full = pd.concat([splits.train, splits.val, splits.test]).reset_index(drop=True)
        boundary = splits.val["time"].max()   # last day of validation window

        X, y, feat_names, times = build_features(full, fc)
        fit_mask = times <= boundary
        test_mask = ~fit_mask

        X_fit, y_fit = X[fit_mask], y[fit_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_test)
        metrics = score_all(y_test, y_pred)

        return TestResult(
            name=candidate.name, metrics=metrics,
            runtime_sec=time.perf_counter() - t0,
            n_features=len(feat_names),
            n_fit=int(fit_mask.sum()), n_test=int(test_mask.sum()),
        )
    except Exception as exc:
        return TestResult(
            name=candidate.name, metrics={},
            runtime_sec=time.perf_counter() - t0,
            n_features=0, n_fit=0, n_test=0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ----------------------------------------------------------------------------
# Test-set spend tracking
# ----------------------------------------------------------------------------

# Column order kept in lockstep with master_log.csv on the metric columns
# (mse_demand, rmse_demand, mae_demand) so the val/test sides are directly
# comparable. Each row of test_log.csv is one (champion x test-eval) event.
TEST_LOG_COLUMNS = [
    "timestamp_utc",
    "champion_name",
    "champion_run_id",
    "val_mse_demand", "val_rmse_demand", "val_mae_demand",
    "test_mse_demand", "test_rmse_demand", "test_mae_demand",
    "test_minus_val_mse", "test_minus_val_pct",
    "n_fit", "n_test",
]


def _read_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=TEST_LOG_COLUMNS)
    return pd.read_csv(log_path)


def _append_log(log_path: Path, row: dict) -> None:
    # Force the canonical column order so an evolving schema doesn't break
    # downstream readers that key off positional columns.
    df = pd.DataFrame([{c: row.get(c) for c in TEST_LOG_COLUMNS}])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def _lookup_val_metrics(champion: dict, master_log_path: Path) -> dict:
    """Pull the champion's val mse/rmse/mae from master_log.csv.

    `champion.json` only stores the primary metric. The full triple
    (mse_demand, rmse_demand, mae_demand) lives in `master_log.csv`,
    keyed by (run_id, candidate_name). Returns whichever fields are
    available; missing ones come back as None and the report shows '—'.
    """
    out = {"mse_demand": float(champion.get("metric")) if champion.get("metric") is not None else None,
           "rmse_demand": None, "mae_demand": None}
    if not master_log_path.exists():
        return out
    try:
        df = pd.read_csv(master_log_path)
    except Exception:
        return out
    mask = df["candidate_name"] == champion["name"]
    if "run_id" in df.columns and champion.get("run_id"):
        mask &= df["run_id"] == champion["run_id"]
    rows = df[mask]
    if not len(rows):
        return out
    # Prefer the row that exactly matches run_id; otherwise take the row
    # with the lowest mse_demand (the "best" reading of this candidate).
    row = rows.sort_values("mse_demand").iloc[0]
    for k in ("mse_demand", "rmse_demand", "mae_demand"):
        if k in row and pd.notna(row[k]):
            out[k] = float(row[k])
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the current champion on the locked test set.",
    )
    parser.add_argument(
        "--champion-path",
        default=f"experiments/auto_runs/{CHAMPION_NAME}",
        help="Path to champion.json (default: experiments/auto_runs/champion.json)",
    )
    parser.add_argument(
        "--results-dir",
        default=str(TEST_DIR_DEFAULT),
        help="Where to write the test-evaluation report + log.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if this champion has already been tested.",
    )
    args = parser.parse_args()

    champion_path = Path(args.champion_path)
    if not champion_path.exists():
        raise SystemExit(
            f"No champion at {champion_path}. Run python run_autoresearch.py first."
        )
    champion = _load_champion(champion_path)
    if champion is None:
        raise SystemExit(f"Empty champion at {champion_path}.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / TEST_LOG_NAME

    log_df = _read_log(log_path)
    prior_evals = (log_df["champion_name"] == champion["name"]).sum() \
        if len(log_df) else 0
    if prior_evals > 0 and not args.force:
        raise SystemExit(
            f"Champion '{champion['name']}' has already been evaluated on the "
            f"test set {prior_evals} time(s). Test-set evaluation is supposed "
            f"to be a one-shot final reading. Pass --force to evaluate again. "
            f"Log: {log_path}"
        )
    if len(log_df):
        print(f"[test-eval] test set has been used {len(log_df)} time(s) "
              f"previously across all champions.")
        print(f"[test-eval] log: {log_path}")

    # ---- Build splits ------------------------------------------------------
    print("[test-eval] loading merged panel + computing locked splits")
    df = load_merged()
    splits = make_splits(df)
    print(splits.describe())
    print()

    # ---- Reconstruct champion + reference baselines ------------------------
    champ_cand = _candidate_from_champion(champion)
    master_log_path = Path(args.champion_path).parent / MASTER_LOG_NAME
    val_metrics = _lookup_val_metrics(champion, master_log_path)
    print(f"[test-eval] champion             : {champ_cand.name}")
    val_msg = ", ".join(
        f"{k}={v:,.2f}" if v is not None else f"{k}=—"
        for k, v in val_metrics.items()
    )
    print(f"[test-eval] champion val metrics : {val_msg}")
    print()

    baselines = baseline_candidates()
    candidates = [champ_cand] + baselines

    # ---- Score every candidate on test ------------------------------------
    results: list[TestResult] = []
    for cand in candidates:
        print(f"[test-eval] scoring on TEST: {cand.name}")
        r = _score_on_test(cand, splits)
        if r.error:
            print(f"  ! error: {r.error}")
        else:
            print(f"  test {PRIMARY_METRIC_NAME}={r.metrics[PRIMARY_METRIC_NAME]:,.2f} "
                  f"  rmse={r.metrics['rmse_demand']:,.2f}  "
                  f"mae={r.metrics['mae_demand']:,.2f}  "
                  f"({r.runtime_sec:.2f}s, n_fit={r.n_fit}, n_test={r.n_test})")
        results.append(r)
    print()

    # ---- Pretty print val vs test side by side ----------------------------
    champ_test = next((r for r in results if r.name == champ_cand.name), None)
    if champ_test is None or champ_test.error:
        raise SystemExit(
            f"Champion failed on test set: {champ_test.error if champ_test else 'no result'}"
        )
    test_metrics = {
        "mse_demand": float(champ_test.metrics["mse_demand"]),
        "rmse_demand": float(champ_test.metrics["rmse_demand"]),
        "mae_demand": float(champ_test.metrics["mae_demand"]),
    }
    val_mse = val_metrics["mse_demand"]
    test_mse = test_metrics["mse_demand"]
    delta = test_mse - val_mse if val_mse is not None else None
    delta_pct = (100 * delta / val_mse) if delta is not None and val_mse else None

    def _fmt(v):
        return f"{v:>16,.2f}" if v is not None else f"{'—':>16}"

    print("=" * 64)
    print("Final test-set readout (validation vs test, side by side)")
    print("=" * 64)
    print(f"  champion : {champ_cand.name}")
    print(f"  {'metric':<12} {'validation':>16} {'test':>16} "
          f"{'test − val':>16}")
    for key in ("mse_demand", "rmse_demand", "mae_demand"):
        v_val = val_metrics[key]
        v_test = test_metrics[key]
        diff = (v_test - v_val) if (v_val is not None and v_test is not None) else None
        print(f"  {key:<12} {_fmt(v_val)} {_fmt(v_test)} {_fmt(diff)}")
    if delta_pct is not None:
        print(f"  test MSE is {delta_pct:+.1f}% relative to validation MSE")
    print()

    print("Reference baselines on TEST (refit on train+val):")
    for r in results:
        if r.error:
            continue
        if r.name in {"seasonal_naive_7", "seasonal_naive_364"}:
            print(f"  {r.name:<22} {PRIMARY_METRIC_NAME}={r.metrics[PRIMARY_METRIC_NAME]:>14,.2f}")
    sn364 = next((r for r in results if r.name == "seasonal_naive_364" and not r.error), None)
    if sn364:
        sn_mse = sn364.metrics[PRIMARY_METRIC_NAME]
        improvement = 100 * (sn_mse - test_mse) / sn_mse
        print(f"  champion is {improvement:+.1f}% relative to seasonal_naive_364 "
              f"on test")
    print()

    # ---- Persist artifact + log ------------------------------------------
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report = {
        "kind": "test_set_evaluation",
        "timestamp_utc": timestamp,
        "champion_name": champion["name"],
        "champion_run_id": champion.get("run_id"),
        # Same metric names as master_log.csv so val and test rows can be
        # joined / compared directly without renaming.
        "champion_val": {
            "mse_demand": val_metrics["mse_demand"],
            "rmse_demand": val_metrics["rmse_demand"],
            "mae_demand": val_metrics["mae_demand"],
        },
        "champion_test": {
            "mse_demand": test_metrics["mse_demand"],
            "rmse_demand": test_metrics["rmse_demand"],
            "mae_demand": test_metrics["mae_demand"],
        },
        "champion_test_minus_val_mse": delta,
        "champion_test_minus_val_pct": delta_pct,
        "splits_summary": splits.describe(),
        "n_fit": champ_test.n_fit,
        "n_test": champ_test.n_test,
        "results": [
            {
                "name": r.name,
                "metrics": r.metrics,
                "runtime_sec": r.runtime_sec,
                "n_features": r.n_features,
                "n_fit": r.n_fit,
                "n_test": r.n_test,
                "error": r.error,
            }
            for r in results
        ],
        "warning": (
            "Test-set evaluation should be a one-shot reading. Each repeat "
            "evaluation reduces the credibility of the test number as a "
            "held-out estimator."
        ),
    }
    out_json = results_dir / f"{champion.get('run_id', 'unknown')}__{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[test-eval] wrote {out_json}")

    _append_log(log_path, {
        "timestamp_utc": timestamp,
        "champion_name": champion["name"],
        "champion_run_id": champion.get("run_id"),
        "val_mse_demand": val_metrics["mse_demand"],
        "val_rmse_demand": val_metrics["rmse_demand"],
        "val_mae_demand": val_metrics["mae_demand"],
        "test_mse_demand": test_metrics["mse_demand"],
        "test_rmse_demand": test_metrics["rmse_demand"],
        "test_mae_demand": test_metrics["mae_demand"],
        "test_minus_val_mse": delta,
        "test_minus_val_pct": delta_pct,
        "n_fit": champ_test.n_fit,
        "n_test": champ_test.n_test,
    })
    print(f"[test-eval] appended row to {log_path}")


if __name__ == "__main__":
    main()
