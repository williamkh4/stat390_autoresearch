# AutoResearch: Victoria Electricity Demand Forecasting

A small, reproducible research framework for the STAT 390 project charter
*"Electricity Grid Demand Forecasting with Weather-Augmented Price Signals
(Victoria, Australia)"*. This repo ships a locked evaluation protocol, a
reference baseline, and an **AutoResearch loop** you can call once per
iteration to score candidate models and track budget/runtime.

---

## Design decisions (locked)

These are fixed for the whole project so that every run is comparable.
Changing any of them invalidates prior leaderboard entries.

| Decision | Value | Where it lives |
|---|---|---|
| Target | Daily `demand` (MW-equivalent) | `src/features.py::TARGET_COL` |
| Validation metric | **Mean Squared Error of demand** | `src/metrics.py::PRIMARY_METRIC_NAME` |
| Test set | **Final 365 days** of the merged panel, locked | `src/split.py::TEST_DAYS` |
| Validation set | 180 days immediately before test | `src/split.py::VAL_DAYS` |
| Train set | All earlier rows | derived |
| Split style | Strict chronological slice (no shuffling) | `src/split.py::make_splits` |
| Baseline | Seasonal naive, `y_hat(t) = demand(t - 7)` | `src/baselines.py::SeasonalNaive` |
| Scope (iter 1) | Single-stage demand prediction | `src/autoresearch.py::default_candidates` |

The two-stage RRP → demand pipeline from the charter is not required for
this iteration; it can be added later as another candidate in the loop.

---

## Repo layout

```
AutoResearch Project/
  data/
    raw/
      open_meteo_data.csv        # raw Open-Meteo export (weather + apparent temp)
      victoria_energy_data.csv   # raw Kaggle Victoria export (demand + RRP)
  README.md                      # this file
  requirements.txt               # pinned minimum versions
  run_baseline.py                # compute baseline MSE on the validation set
  run_autoresearch.py            # one full pass of the AutoResearch loop
  src/
    __init__.py
    data_loader.py               # load + merge the two CSVs
    split.py                     # locked train/val/test splitter
    metrics.py                   # MSE/RMSE/MAE, with MSE marked PRIMARY
    features.py                  # FeatureConfig + build_features
    baselines.py                 # SeasonalNaive
    autoresearch.py              # Candidate registry + run_loop()
  experiments/
    results/                     # run_<id>.json + leaderboard CSVs land here
```

---

## Setup

```bash
# (Optional) fresh virtual env
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Verify the data files are in place:

```
data/raw/open_meteo_data.csv
data/raw/victoria_energy_data.csv
```

The loader (`src/data_loader.py`) defaults to `<repo root>/data/raw/`, so
no further configuration is needed.

---

## 1. Baseline analysis

Computes the locked validation metric (MSE of demand) for every baseline
registered in `default_candidates()`. Two baselines ship by default:

- `seasonal_naive_7` — `y_hat(t) = demand(t - 7)`. Weekly-cycle floor.
- `seasonal_naive_364` — `y_hat(t) = demand(t - 364)`. Same day-of-week,
  52 weeks ago; captures the annual cycle trivially. The strongest baseline
  is reported as the "number to beat".

```bash
python run_baseline.py
```

Artifacts (relative to the current working directory; run from the repo root):

```
experiments/results/baseline_<timestamp>.json    # full report
experiments/results/baseline_<timestamp>.csv     # leaderboard CSV
```

If you re-run the script and don't see new files, check (1) your current
working directory when you invoked `python`, and (2) `experiments/results/`
under that directory. The script prints the exact path of every file it
writes on its last two lines.

---

## 2. Run the AutoResearch loop once

One iteration of the loop fits every candidate in
`src/autoresearch.py::default_candidates()` on the train split, scores each
on validation, and writes a leaderboard plus a full JSON run report.

```bash
python run_autoresearch.py
```

Artifacts (relative to the current working directory; run from the repo root):

```
experiments/results/run_<id>.json              # full report: per-candidate metrics, env, splits
experiments/results/run_<id>_leaderboard.csv   # sortable CSV leaderboard
```

`<id>` is a fresh 8-character UUID per run, so consecutive runs never
overwrite each other.

At the end of the run you will see a budget summary like:

```
Budget summary
  wall clock (including data load): 4.21 s
  loop-only runtime:                3.97 s
  model fits (budget units):        5
  avg runtime per fit:              0.79 s
  run_id:                           a1b2c3d4
```

Use the leaderboard to decide what to change next (new features, new models,
different hyperparameters) and add them to `default_candidates()` for the
next iteration.

---

## Reproducibility notes

- **No randomness in the split.** The test and val windows are defined by
  row count from the end of the merged panel, so the same CSVs produce the
  same splits every time.
- **Fixed model seeds.** Ridge / RF / GBM candidates pass `random_state=0`.
  Pinning `numpy` and `scikit-learn` versions via `requirements.txt` keeps
  results stable across machines.
- **Test set is inaccessible to the loop.** `run_loop` consumes only
  `Splits.train` and `Splits.val`; `Splits.test` is constructed but never
  passed to any model until a deliberate final-evaluation step (not part of
  this iteration).
- **Runtime is measured with `time.perf_counter()` inside the loop**, so
  the reported numbers exclude import cost. The outer wall clock in
  `run_autoresearch.py` additionally includes data loading.

---

## Extending the loop

To add a new candidate, append to `default_candidates()` in
`src/autoresearch.py`:

```python
Candidate(
    name="my_new_idea",
    feature_config=FeatureConfig(
        use_calendar=True, use_temp=True, use_apparent_temp=True,
        use_rrp=False, demand_lags=[1, 7, 14], rolling_windows=[7, 28],
    ),
    model_factory=lambda: MyEstimator(...),
),
```

Re-run `python run_autoresearch.py` and compare the leaderboard to the
previous `run_<id>_leaderboard.csv`. Budget (fit count) and runtime give
you a cost-vs-improvement picture as the search space grows.
