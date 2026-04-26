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
  analyze_runs.py                # cross-run leaderboard / champion / ablation tool
  src/
    __init__.py
    data_loader.py               # load + merge the two CSVs
    split.py                     # locked train/val/test splitter
    metrics.py                   # MSE/RMSE/MAE, with MSE marked PRIMARY
    features.py                  # FeatureConfig + build_features
    baselines.py                 # SeasonalNaive
    numpy_models.py              # NumpyOLS (sklearn-free linear baseline)
    autoresearch.py              # Candidate registry, MODEL_SPECS, FEATURE_PRESETS, run_loop()
  experiments/
    baseline_runs/               # baseline_<timestamp>.json + .csv from run_baseline.py
    auto_runs/                   # run_<id>.json, leaderboards, master_log.csv, champion.json
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
experiments/baseline_runs/baseline_<timestamp>.json    # full report
experiments/baseline_runs/baseline_<timestamp>.csv     # leaderboard CSV
```

If you re-run the script and don't see new files, check (1) your current
working directory when you invoked `python`, and (2) `experiments/baseline_runs/`
under that directory. The script prints the exact path of every file it
writes on its last two lines.

---

## 2. Run the AutoResearch loop once (self-driving)

`python run_autoresearch.py` is **self-driving** -- you do not edit any code
between iterations. Each invocation:

1. Reads the current champion from `experiments/auto_runs/champion.json`
   (None on the first run).
2. Reads `experiments/auto_runs/master_log.csv` for the set of candidate names
   already tried.
3. Calls `auto_candidates()` (in `src/autoresearch.py`) which assembles:
   - both seasonal-naive **baselines** (always);
   - the **champion's exact config** (always, so noise on the current
     leader is visible in the new run's leaderboard);
   - **`--n-challengers`** new challengers (default 4), prioritising
     **one-knob mutations of the champion's feature config** before
     falling back to random untried points from the full search space.
4. Fits + scores all of those, writes the artifacts, appends to
   `master_log.csv`, and auto-promotes a new champion if any challenger
   beat the previous one.

Because step 2 dedupes against history, every run automatically does *new*
work without you re-editing the candidate list.

```bash
python run_autoresearch.py                       # default 4 challengers
python run_autoresearch.py --n-challengers 6     # try more this run
python run_autoresearch.py --seed 42             # reproducible draw
```

The discrete search space lives in `src/autoresearch.py`:

- `MODEL_SPECS` -- model_type x kwargs (numpy OLS variants, sklearn Ridge,
  Random Forest, Gradient Boosting). sklearn-dependent specs are silently
  skipped if sklearn isn't installed.
- `FEATURE_PRESETS` -- short tag plus a `FeatureConfig` (calendar, temp,
  apparent_temp, RRP, demand_lags, rolling_windows). Each preset combines
  with each model spec to form one candidate; names are deterministic, e.g.
  `ridge__cal_temp_apptemp_lag1-7_roll7__alpha1.0`.

To enlarge the search space, append to `MODEL_SPECS` or `FEATURE_PRESETS`
and re-run -- new candidates will surface organically as challengers.

Artifacts (relative to the current working directory; run from the repo root):

```
experiments/auto_runs/run_<id>.json              # full report: per-candidate metrics, env, splits
experiments/auto_runs/run_<id>_leaderboard.csv   # sortable CSV leaderboard
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

The loop maintains two cross-run files in `experiments/auto_runs/`:

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

You don't have to. The auto-generator already prioritises **one-knob
mutations of the current champion's feature config**, which is the natural
ablation pair (same model class, one feature toggled). Examples of pairs the
generator will surface automatically over consecutive runs:

| comparison | what it isolates |
|---|---|
| `ridge__cal_temp__alpha1.0` vs `ridge__cal_temp_apptemp__alpha1.0` | apparent-temperature feature |
| `ridge__cal_temp_apptemp_lag1-7_roll7__alpha1.0` vs same with `__alpha10.0` | regularisation strength |
| `numpy_ols__full__alpha1.0` vs `ridge__full__alpha1.0` | numpy OLS vs sklearn Ridge |
| any candidate vs `seasonal_naive_364` | "did I beat the trivial yearly recall?" |

Decision rule (used implicitly): the run promotes whatever has the lowest
validation MSE to champion. Use `python analyze_runs.py --ablation` to
inspect the per-candidate mean/std once you have multiple runs, and to
manually compare smallest-meaningfully-better candidates if runtime cost
matters.

To **steer** the search rather than let it drift -- e.g. spend the next run
exclusively on tree models -- temporarily delete `experiments/auto_runs/
champion.json` (so the auto-generator has no "champion mutations" to draw
from and falls back to random untried points), or pass a specific
`--seed` and re-run until the desired candidates show up. Permanent
steering should happen by editing `MODEL_SPECS`/`FEATURE_PRESETS` rather
than by mutating individual run scripts.

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

To add a new model class or hyperparameter point, append a tuple to
`MODEL_SPECS` in `src/autoresearch.py`:

```python
MODEL_SPECS.append(
    ("ridge", (("alpha", 100.0), ("random_state", 0))),
)
```

To add a new feature preset, append to `FEATURE_PRESETS`:

```python
FEATURE_PRESETS.append((
    "cal_temp_apptemp_lag1-7-14-30_roll7-28",
    FeatureConfig(
        use_calendar=True, use_temp=True, use_apparent_temp=True,
        use_rrp=False, demand_lags=[1, 7, 14, 30], rolling_windows=[7, 28],
    ),
))
```

If the new model type is something other than the built-in `numpy_ols`,
`ridge`, `rf`, `gbm`, also add a branch in `_make_factory()`. Then re-run
`python run_autoresearch.py`. New combos will surface as challengers in
subsequent runs without any further manual selection.

Budget (fit count) and runtime in each run report give a cost-vs-improvement
picture as the search space grows.
