"""
Chronological train / validation / test splits.

Contract (locked for this project):

  * TEST  = final 365 days of the merged panel. Never touched inside the
    AutoResearch loop. Only used for the final held-out evaluation.

  * Iter-1 "holdout" protocol -- single split:
      VAL   = the 180 days immediately preceding the test window.
      TRAIN = everything before VAL.

  * Iter-2 "walk-forward" protocol -- many folds:
      For fold i (i = 0, 1, 2, ...):
        train_end_i = min_train_size + i * step    (in days, from start)
        val_window_i = the next `val_size` days
      Train expands with i (it does NOT roll), val shifts forward, the
      locked test window is the same for every fold. Folds whose val
      window would overlap the locked test window are dropped.

Both protocols return strict chronological slices (no shuffling) because
the project brief flags time leakage as a top risk and MSE on a random
split would be optimistic about a model that has seen the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd


# ---- Holdout (iter-1) -----------------------------------------------------

TEST_DAYS = 365
VAL_DAYS = 180


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def window(df):
            return f"{df['time'].min().date()} -> {df['time'].max().date()} ({len(df)} rows)"
        return (
            f"train: {window(self.train)}\n"
            f"val:   {window(self.val)}\n"
            f"test:  {window(self.test)}  [LOCKED]"
        )


def make_splits(
    df: pd.DataFrame,
    test_days: int = TEST_DAYS,
    val_days: int = VAL_DAYS,
    time_col: str = "time",
) -> Splits:
    """Return train/val/test slices in chronological order.

    The input must be sorted ascending by `time_col`; we re-sort defensively.
    """
    if time_col not in df.columns:
        raise ValueError(f"Expected '{time_col}' column in dataframe")

    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    if n <= test_days + val_days:
        raise ValueError(
            f"Not enough rows ({n}) for test={test_days} + val={val_days} split"
        )

    test_start = n - test_days
    val_start = test_start - val_days

    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    return Splits(train=train, val=val, test=test)


# ---- Walk-forward (iter-2) -----------------------------------------------

# Default walk-forward parameters. These are locked decisions in iter-2,
# changing them invalidates prior master_log entries for the walk-forward
# protocol the same way the iter-1 hold-out parameters do.
WF_VAL_SIZE = 180
WF_STEP = 90
WF_MIN_TRAIN = 730


@dataclass(frozen=True)
class WalkForwardFold:
    """One fold of the expanding-window walk-forward CV.

    `train` expands with `fold_idx`; `val` slides forward by `WF_STEP`
    days each fold; `test` is the locked final 365 days and is the same
    object on every fold (held back, never read by the auto loop).
    """
    fold_idx: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def window(df):
            return (f"{df['time'].min().date()} -> "
                    f"{df['time'].max().date()} ({len(df)} rows)")
        return (
            f"fold {self.fold_idx}:\n"
            f"  train: {window(self.train)}\n"
            f"  val:   {window(self.val)}\n"
            f"  test:  {window(self.test)}  [LOCKED]"
        )


def make_walk_forward_folds(
    df: pd.DataFrame,
    val_size: int = WF_VAL_SIZE,
    step: int = WF_STEP,
    min_train_size: int = WF_MIN_TRAIN,
    test_days: int = TEST_DAYS,
    time_col: str = "time",
) -> List[WalkForwardFold]:
    """Expanding-window walk-forward folds.

    Locked test window (`test_days`) sits at the end of the panel and is
    carried as a constant reference on every fold (the auto loop never
    reads it, but downstream code occasionally needs the rows to compute
    lag-aligned features without breaking the time series).

    Returns ~10 folds for the project's ~2,106-row panel under the
    defaults (val_size=180, step=90, min_train=730, test_days=365):
    val window endings march from ~mid-2017 to ~mid-2019.

    Algorithm:
      val_start = min_train_size + i * step          (fold i, 0-indexed)
      val_end   = val_start + val_size
      train      = rows[: val_start]
      val        = rows[val_start : val_end]
      test       = rows[-test_days :]                (same every fold)

    A fold is kept only if `val_end <= n - test_days` (i.e. the val
    window does not bleed into the locked test window). The next fold's
    `val_start` is `step` days later, so adjacent val windows overlap by
    `val_size - step = 90` days (50% overlap, deliberately, to amortise
    the cost of a fit across more readings; the "independent" subset is
    every-other fold).
    """
    if time_col not in df.columns:
        raise ValueError(f"Expected '{time_col}' column in dataframe")
    if val_size <= 0 or step <= 0 or min_train_size <= 0:
        raise ValueError("val_size, step, min_train_size must all be > 0")

    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    if n <= test_days + min_train_size + val_size:
        raise ValueError(
            f"Not enough rows ({n}) for walk-forward: need "
            f"min_train={min_train_size} + val={val_size} + test={test_days}."
        )

    test = df.iloc[n - test_days:].copy()
    last_val_end_allowed = n - test_days   # exclusive

    folds: List[WalkForwardFold] = []
    i = 0
    while True:
        val_start = min_train_size + i * step
        val_end = val_start + val_size
        if val_end > last_val_end_allowed:
            break
        train = df.iloc[:val_start].copy()
        val = df.iloc[val_start:val_end].copy()
        folds.append(WalkForwardFold(fold_idx=i, train=train, val=val, test=test))
        i += 1

    if not folds:
        raise RuntimeError(
            "Zero walk-forward folds produced. Check that "
            "min_train + val_size + test_days <= len(df)."
        )
    return folds


def walk_forward_summary(folds: List[WalkForwardFold]) -> str:
    """Pretty-print a single-line summary of a walk-forward design."""
    if not folds:
        return "0 folds"
    f0, fN = folds[0], folds[-1]
    return (
        f"walk-forward: {len(folds)} folds, "
        f"val window {len(f0.val)} days, "
        f"step ~{(folds[1].val['time'].min() - f0.val['time'].min()).days if len(folds) > 1 else 0} days, "
        f"train expands from {len(f0.train)} to {len(fN.train)} rows, "
        f"first val starts {f0.val['time'].min().date()}, "
        f"last val ends {fN.val['time'].max().date()}, "
        f"test [LOCKED] {f0.test['time'].min().date()} -> {f0.test['time'].max().date()}"
    )


def pre_covid_test_window(
    df: pd.DataFrame,
    test_days: int = TEST_DAYS,
    time_col: str = "time",
) -> pd.DataFrame:
    """Return the 365 days ending one day before the locked test window.

    Used by `run_test_evaluation.py --pre-covid-sensitivity` to separate
    "model quality" from "COVID regime change" in the final readout.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    if n < 2 * test_days:
        raise ValueError(
            f"Not enough rows ({n}) for pre-COVID window: need >= {2 * test_days}."
        )
    pre_end = n - test_days       # exclusive of test
    pre_start = pre_end - test_days
    return df.iloc[pre_start:pre_end].copy()


if __name__ == "__main__":
    from src.data_loader import load_merged

    df = load_merged()
    print("=== holdout splits (iter-1 protocol) ===")
    print(make_splits(df).describe())
    print()
    print("=== walk-forward folds (iter-2 protocol) ===")
    folds = make_walk_forward_folds(df)
    print(walk_forward_summary(folds))
    print()
    for f in folds:
        print(f.describe())
        print()
    print("=== pre-COVID sensitivity window ===")
    pc = pre_covid_test_window(df)
    print(f"{pc['time'].min().date()} -> {pc['time'].max().date()} ({len(pc)} rows)")
