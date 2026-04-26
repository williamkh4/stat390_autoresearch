"""
Load and merge the two raw data sources for the project:

  - victoria_energy_data.csv  (Kaggle Victoria: demand, RRP, temp, holiday, school_day)
  - open_meteo_data.csv       (Open-Meteo: temperature + apparent temperature + sun/rain)

Both files are daily for Victoria, Australia.

The Open-Meteo export has a 3-line metadata header (lat/lon/elev + blank line)
before the actual time-series header, so we parse it defensively.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# Default location: <repo root>/data/raw/, matching the charter's repo layout.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _find_timeseries_header(path: Path) -> int:
    """Return the 0-indexed line number of the row that starts with 'time,'."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("time,"):
                return i
    raise ValueError(f"Could not find a 'time,' header row in {path}")


def load_victoria_energy(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the Kaggle Victoria daily demand/RRP dataset."""
    path = Path(data_dir) / "victoria_energy_data.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"date": "time"})
    df = df.sort_values("time").reset_index(drop=True)
    # Normalize categorical flags to 0/1.
    df["school_day"] = (df["school_day"].astype(str).str.upper() == "Y").astype(int)
    df["holiday"] = (df["holiday"].astype(str).str.upper() == "Y").astype(int)
    return df


def load_open_meteo(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the Open-Meteo daily weather export, skipping metadata preamble."""
    path = Path(data_dir) / "open_meteo_data.csv"
    header_row = _find_timeseries_header(path)
    df = pd.read_csv(path, skiprows=header_row, parse_dates=["time"])
    # Simplify unit-laden column names to snake_case.
    rename = {
        "temperature_2m_mean (°C)": "temp_mean",
        "temperature_2m_max (°C)": "temp_max",
        "temperature_2m_min (°C)": "temp_min",
        "apparent_temperature_mean (°C)": "apparent_temp_mean",
        "apparent_temperature_max (°C)": "apparent_temp_max",
        "apparent_temperature_min (°C)": "apparent_temp_min",
        "sunshine_duration (s)": "sunshine_s",
        "rain_sum (mm)": "rain_mm",
        "precipitation_sum (mm)": "precip_mm",
        "snowfall_sum (cm)": "snowfall_cm",
        "daylight_duration (s)": "daylight_s",
    }
    df = df.rename(columns=rename).sort_values("time").reset_index(drop=True)
    return df


def load_merged(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Return the merged daily panel with demand as the target."""
    energy = load_victoria_energy(data_dir)
    weather = load_open_meteo(data_dir)
    merged = pd.merge(energy, weather, on="time", how="inner").sort_values("time")
    merged = merged.reset_index(drop=True)
    # Guard against duplicate dates (should be none at daily resolution).
    assert merged["time"].is_unique, "Duplicate dates in merged panel"
    return merged


def validate_merged(df: pd.DataFrame) -> List[str]:
    """Check the merged panel for silent data-quality issues.

    Returns a list of human-readable warning strings (empty if everything
    looks clean). Surfaces the failure modes documented in the README:
      - missing or duplicate dates
      - gaps in the date sequence
      - NaNs in the target column (`demand`)
      - low row count after merge (suggests poor source overlap)
    """
    warnings: List[str] = []
    if df.empty:
        return ["Merged panel is empty -- no overlap between sources."]

    if not df["time"].is_unique:
        warnings.append(
            f"Duplicate dates in merged panel: "
            f"{df['time'].duplicated().sum()} rows."
        )

    full_range = pd.date_range(df["time"].min(), df["time"].max(), freq="D")
    missing = full_range.difference(df["time"])
    if len(missing):
        warnings.append(
            f"{len(missing)} dates missing from the contiguous range "
            f"({df['time'].min().date()} -> {df['time'].max().date()}). "
            f"First gaps: {[str(d.date()) for d in missing[:5]]}"
        )

    if df["demand"].isna().any():
        warnings.append(
            f"`demand` has {int(df['demand'].isna().sum())} NaN(s); "
            "these rows will be dropped during feature build."
        )

    if len(df) < 1000:
        warnings.append(
            f"Merged panel has only {len(df)} rows -- expected ~2,100. "
            "Check that both source CSVs cover the same date range."
        )

    return warnings


# Need to import List for the validator.
from typing import List  # noqa: E402  (placed late to keep public API at top)


if __name__ == "__main__":
    df = load_merged()
    print(f"Merged rows:    {len(df):,}")
    print(f"Date range:     {df['time'].min().date()}  ->  {df['time'].max().date()}")
    print(f"Columns:        {list(df.columns)}")
    print(df.head(3))

    print()
    print("Data quality warnings:")
    warns = validate_merged(df)
    if warns:
        for w in warns:
            print(f"  - {w}")
    else:
        print("  none -- panel looks clean.")
