"""
Chronological train / validation / test split.

Contract (locked for this project):

  * TEST  = final 365 days of the merged panel. Never touched inside the
    AutoResearch loop. Only used for the final held-out evaluation.
  * VAL   = the 180 days immediately preceding the test window. This is the
    metric surface the AutoResearch loop optimizes against.
  * TRAIN = everything before VAL.

Using calendar-aligned slices (not random shuffles) is essential: the brief
flags "time leakage" as a top risk, and MSE on a random split would be
optimistic about a model that has seen the future.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


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


if __name__ == "__main__":
    from src.data_loader import load_merged

    splits = make_splits(load_merged())
    print(splits.describe())
