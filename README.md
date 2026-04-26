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

## 3. Cross-run analysis & champion tracking

The loop maintains two cross-run files in `experiments/results/`:

- `master_log.csv` — every candidate from every run, appended one row each.
  This is your single source of truth for "what have I tried so far?".
- `champion.json` — the current best-performing candidate. Auto-updated
  whenever a run produces a result with a lower validation MSE than the
  previous champion. Each run prints "current champion" at the start and
  "NEW CHAMPION" / "champion unchanged" at the end.

To inspect history without writing pandas every time:

```bash
python analyze_runs.py             # all-time leaderboard + per-run summary + champion
python analyze_runs.py --top 20    # show more leaderboard rows
python analyze_runs.py --ablation  # mean / std / best per candidate across runs
```

### Picking specs for the next iteration

Use the cross-run leaderboard to spot **ablation pairs** — candidates that
differ by exactly one knob:

| comparison | what it isolates |
|---|---|
| `ridge_cal_temp` vs `ridge_cal_temp_apptemp` | apparent-temperature feature |
| `ridge_cal_temp_apptemp` vs `rf_cal_temp_apptemp` | linearity vs trees |
| `rf_cal_temp_apptemp` vs `gbm_full` | adding observed RRP + boosting |
| any candidate vs `seasonal_naive_364` | "did I beat the trivial yearly recall?" |

Decision rule: pick the *smallest* candidate whose MSE meaningfully beats
the strongest baseline; ties broken by lower `runtime_sec`. Then set up
your next iteration by adding new variants of that candidate (different
hyperparameters, more lags, different features) to `default_candidates()`
and re-running.

The champion is the natural reference for the next iteration: any new
candidate you add should aim to beat it. Because it auto-updates, you
never have to manually re-anchor.

---

## Failure modes & how to see them

### Silent (the dangerous ones)

These don't crash the run; they quietly distort results.

| Failure | How it shows up | How to detect |
|---|---|---|
| Date gaps in merged panel | Lag features computed across gaps as if they were contiguous | `python -m src.data_loader` runs the validator and lists warnings |
| Non-overlapping source date ranges | Merged panel shrinks; `n_train` in run report drops | Validator + `n_train` in `run_<id>.json` |
| `demand` NaNs | Rows silently dropped during feature build | Validator + comparing `n_train` across candidates |
| Lag warmup eats early train rows | `n_train` smaller than expected | `n_train` in run report; expected ≈ rows minus longest lag |
| Day-of-week shift via `period=365` (non-multiple of 7) | Slightly worse MSE for what looks like a yearly cycle | Compared `period=364` vs `period=365` explicitly; baseline uses 364 |
| Observed RRP used as feature (semantic) | Optimistic MSE for `gbm_full` since real-time you'd only have *predicted* RRP | Flagged here; future iteration should swap to a 2-stage pipeline (charter calls for this) |
| SeasonalNaive lookup falls back to mean | Flat predictions at the start of validation | Only happens if `period > train history depth`; not the case here |
| Train/val/test config changed mid-project | All prior runs become incomparable | `splits_summary` is recorded in every `run_<id>.json` — diff it against past runs |
| Concurrent runs racing on `master_log.csv` | Interleaved/torn rows | Avoid: don't run two loops in parallel against the same `--results-dir` |

### Loud (the visible ones)

These either crash or show clearly in the artifacts.

| Failure | Where you see it |
|---|---|
| Missing data CSV | `FileNotFoundError` from `data_loader.py` at startup |
| Wrong CWD when running scripts | Files land where you ran from, not project root; check the "Wrote: ..." lines printed by each script |
| A candidate raises during fit/predict | `error` field is populated for that row in `run_<id>_leaderboard.csv`, `master_log.csv`, and the JSON; the loop continues with the rest |
| `sklearn` not installed | Import error from `default_candidates()`. The baseline path (`run_baseline.py`) does NOT need sklearn |
| Duplicate dates in source CSV | `assert merged["time"].is_unique` fails in `data_loader.load_merged` |

### Surfacing failures

Three artifacts you should look at after any run:

1. `run_<id>.json` → `splits_summary`, `env`, per-candidate `error` and `n_train`/`n_val`/`n_features` (silent-issue indicators).
2. `master_log.csv` → quickly grep for `error` non-empty.
3. `python analyze_runs.py` → if a candidate disappears from the leaderboard you didn't expect, it errored out.

Run `python -m src.data_loader` whenever data files change to re-check silent data-quality issues before kicking off another loop.

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
