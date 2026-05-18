# Program Specification — Victoria Electricity Demand Forecasting (AutoResearch)

| Field | Value |
|---|---|
| Project | Electricity Grid Demand Forecasting with Weather-Augmented Price Signals |
| Region | Victoria, Australia |
| Time resolution | Daily |
| Spec version | v0.2 (iteration 2, two-stage + walk-forward) |
| Owner | Will Huang |
| Source brief | STAT 390 Week 1 Project Charter |

---

## 1. Research question

> **Research question (v0.2).** Given iteration 1's finding that
> apparent-temperature features and demand lags alone yield a validation
> MSE of ~9M on Victoria daily electricity demand (an 81% reduction
> relative to the strongest seasonal-naive baseline), this iteration
> tests whether a **two-stage pipeline** — predicting daily RRP from
> weather + calendar features in stage 1, then using the *predicted* RRP
> as an input to the demand model in stage 2 — produces a forecast MSE
> that is (a) lower than the single-stage temperature+lags baseline by a
> margin larger than the bootstrap-derived noise floor, and (b) survives
> a sensitivity check against a 365-day pre-COVID test window.
>
> **Validation protocol.** Expanding-window walk-forward cross-validation.
> Per fold: `val_size = 180` days, `step = 90` days (quarterly cadence),
> `min_train_size = 730` days (≥1 annual cycle), train expands with each
> fold, never rolls. Approximately 10 folds span 2017-Q1 through 2019-Q3.
> Adjacent val windows overlap by 50%; the leaderboard reports `mean ± std`
> across all 10 folds AND `std_indep` across the 5 non-overlapping subset
> (every-other fold). Champion promotion is noise-aware: a challenger is
> promoted only if `mean_challenger + std_challenger
> < mean_champion − std_champion`.
>
> **Test-set protocol.** The locked 365-day test window
> (2019-10-08 → 2020-10-06) is evaluated **exactly once** at the end of
> iteration 2, via `run_test_evaluation.py`. A 365-day **pre-COVID
> sensitivity** window (the 365 days ending 2019-10-07) is also evaluated
> once, alongside the locked test, to separate "model quality" from
> "COVID-era regime change."
>
> **Project contribution.** Three claims this iteration commits to
> producing:
> 1. **An honest H2 test.** Whether *predicted* RRP — via a two-stage
>    pipeline — adds signal beyond temperature + demand history. Either
>    direction (supported / null / rejected) is a publishable result; an
>    explicit minimum-effect-size of 5% relative MSE reduction is
>    required to call H2 "supported."
> 2. **A methodology artifact.** An auto-research framework with
>    walk-forward CV, bootstrap noise floor, integrated four-category
>    error taxonomy (Signal / Code / Eval-Leakage / Agent), and a
>    self-driving candidate generator with history-aware dedup.
>    Demonstrates iterative model selection on a small dataset without
>    overfitting comparisons.
> 3. **A distribution-shift finding.** Quantification of the ~3×
>    degradation between val MSE and locked-test MSE (deterministic for
>    the seasonal-naive baselines), with the pre-COVID sensitivity
>    readout decomposing "model quality" from "COVID regime change."

---

## 2. Hypotheses

1. **H1 (apparent temperature):** Including Open-Meteo's apparent-temperature
   features lowers validation MSE compared to the same model with only
   standard temperature features. **Status: supported in iter-1; locked,
   not retested.** Iter-1 champion uses apparent_temp; ablation pair A2→A3
   in the controlled-experiments bundle shows the gain is real (val MSE
   10.07M → 9.49M, ~5.7% reduction).
2. **H2 (price signal):** Adding *predicted* RRP — via the two-stage
   pipeline in `src/predict_rrp.py` — lowers validation MSE versus the
   same model without RRP. **Status: primary iter-2 work; minimum
   effect size 5% relative MSE reduction is required to call H2
   "supported."** Iter-1's `full` preset used observed RRP and was leaky
   (L1 in `ERROR_TAXONOMY.md`); the predicted-RRP variant in iter-2
   replaces it and is testable cleanly.
3. **H3 (non-linearity):** Tree ensembles outperform linear models even
   with the same feature set, because demand-temperature interactions are
   non-linear (e.g., U-shaped: heating below ~12 °C, cooling above ~24 °C).
   **Status: supported in iter-1; locked, not retested.** Iter-1 result
   matrix shows trees beating linear by ~4× across every preset (best
   MLP 8.88M vs best Ridge 41.12M); linear models are retained in iter-2
   only as sanity-only rows.
4. **M1 (methodology, new):** Walk-forward fold MSEs cluster within a
   band predicted by the bootstrap noise floor — i.e. our reported
   `std` is calibrated. If the across-run variance of the champion's
   per-fold MSEs is much larger than the within-run `std`, then `std`
   is under-reporting the true noise and the noise-aware promotion rule
   is mis-tuned. M1 is an internal calibration check, not a research
   question about demand forecasting.

---

## 3. Scope

**In scope (iter-2):**
- **Two-stage RRP → demand pipeline.** Stage-1 (`src/predict_rrp.py`) fits
  a GBM on weather + calendar + RRP lags to produce a predicted RRP
  column; stage-2 (the demand model) consumes that column via
  `FeatureConfig.use_predicted_rrp`. No demand features in stage 1.
- **Walk-forward cross-validation.** Expanding-window, 10 folds,
  val_size=180, step=90, min_train=730. `mean ± std` reporting per
  metric; `std_indep` reported across the 5 non-overlapping folds.
- **Bootstrap noise floor.** The per-fold `std` plays the role iter-1
  lacked (deterministic `std=0` made every leaderboard gap look real).
  All iter-2 comparisons are noise-aware.
- **Noise-aware champion promotion** — see §5 row "Champion promotion
  rule."
- **Pre-COVID sensitivity readout** (one-shot, alongside the locked test).
- **Search-space narrowing** based on iter-1 result-matrix evidence — see
  `ABLATION_TABLES.md`.

**Out of scope (iter-2, deferred):**
- Deep learning beyond MLP (LSTM, Transformer).
- Sub-daily resolution (intraday, half-hourly).
- Probabilistic forecasts / prediction intervals.
- Stage-1 RRP model-class sweeps (use the GBM default; if it
  underperforms, that's a finding, not a sub-project).
- A separate generalization-under-COVID study beyond the single
  pre-COVID sensitivity readout.
- Repeated random search over the full 494-candidate space; iter-2's
  search space is the narrowed primary set in `ABLATION_TABLES.md`.

---

## 4. Data

| Source | File | Description |
|---|---|---|
| Kaggle Victoria | `data/raw/victoria_energy_data.csv` | Daily demand, RRP, min/max temperature, solar exposure, rainfall, school-day & holiday flags, 2015–2020 |
| Open-Meteo API | `data/raw/open_meteo_data.csv` | Daily temperature mean/max/min, apparent temperature mean/max/min, sunshine, rain/precip/snowfall, daylight, 2015–2020 |

Merged panel (after inner join on date): **2,106 daily rows**, 2015-01-01 → 2020-10-06, 25 columns. No missing dates, no NaNs in `demand` (verified by `src/data_loader.validate_merged`).

Licensing: Kaggle dataset is typically CC0; Open-Meteo data is CC BY 4.0
(attribution required if redistributed). README includes attribution notes.

---

## 5. Locked design decisions

These are fixed for the duration of the project. Changing any of them
invalidates prior `master_log.csv` entries and resets the champion.

| Decision | Value | Source of truth |
|---|---|---|
| Target | Daily `demand` | `src/features.py::TARGET_COL` |
| Validation metric | MSE of demand | `src/metrics.py::PRIMARY_METRIC_NAME` |
| Test set | Final 365 days, never touched in the loop | `src/split.py::TEST_DAYS` |
| Validation set (iter-1 holdout) | 180 days immediately before test | `src/split.py::VAL_DAYS` |
| Train set | All earlier rows | derived |
| Split style | Strict chronological slice | `src/split.py::make_splits` |
| Baselines | `seasonal_naive_7` (weekly floor) and `seasonal_naive_364` (yearly floor; 52-week aligned) | `src/autoresearch.py::baseline_candidates` |
| Reference for new candidates | Current champion in `experiments/auto_runs/champion.json`, auto-promoted on improvement | `src/autoresearch.py::run_loop` |
| **Validation protocol (iter-2)** | Walk-forward expanding-window CV: val_size=180, step=90, min_train=730 (~10 folds, train expands, never rolls) | `src/split.py::make_walk_forward_folds` |
| **Champion promotion rule (iter-2)** | Noise-aware: `mean_challenger + std_challenger < mean_champion − std_champion` | `src/autoresearch.py::_noise_aware_promote` |
| **Two-stage RRP (iter-2)** | Stage-1 RRP predictor (GBM-300/3/0.05) fit on weather + calendar + RRP lags only (no demand features); output column `rrp_predicted` consumed by stage-2 demand model via `FeatureConfig.use_predicted_rrp = True` | `src/predict_rrp.py` |
| **Observed-RRP feature (iter-2)** | **Deprecated.** Existing candidates using observed RRP are marked `uses_observed_rrp=True` in master_log and excluded from new champion comparisons. | `src/features.py::FeatureConfig.use_rrp` |
| **Pre-COVID sensitivity (iter-2)** | One-shot 365-day window ending one day before locked test, evaluated alongside the locked test in `run_test_evaluation.py --pre-covid-sensitivity` | `src/split.py::pre_covid_test_window` |

---

## 6. Methodology

```
Raw CSVs
  └─> data_loader.load_merged()   (with validate_merged sanity check)
       └─> split.make_splits()    (train | val | test, locked)
            └─> features.build_features(FeatureConfig)
                 └─> Candidate.model_factory().fit() / .predict()
                      └─> metrics.score_all()   (MSE primary, RMSE/MAE secondary)
                           └─> autoresearch.run_loop()
                                ├─> per-run JSON + leaderboard CSV
                                ├─> append to master_log.csv
                                └─> update champion.json on improvement
```

Feature configuration is per-candidate, exposed through `FeatureConfig` so
that feature-set choice is part of the search space rather than hard-coded
in any one script. Lag and rolling features are computed via `.shift()` so
they can never look forward; rows where lags would reach past the start of
the panel are dropped.

---

## 7. Success criteria

1. **Floor (iter-2):** at least one non-baseline candidate satisfies
   `mean − std  <  seasonal_naive_364.mean − seasonal_naive_364.std`
   under the walk-forward protocol. A model that doesn't beat trivial
   yearly recall under the noise-aware rule is not adding research
   value.
2. **H1 acceptance:** Direct ablation pair (same model class, same other
   features) shows that adding apparent-temperature features lowers
   walk-forward MSE by a margin larger than the per-fold std.
   **Status: met in iter-1's controlled experiments; locked, not retested.**
3. **H2 acceptance (iter-2 primary criterion):** the predicted-RRP
   variant of the current champion beats the no-RRP variant of the same
   champion by **≥5% relative MSE reduction** *and* the noise-aware
   rule fires:
   `mean_pred-rrp + std_pred-rrp < mean_no-rrp − std_no-rrp`.
   Anything weaker is reported as "H2 not supported under the iter-2
   minimum-effect-size criterion."
4. **M1 acceptance (methodology):** for the iter-1 champion replayed
   under walk-forward, the across-run variance of the per-fold mean MSE
   is comparable to (or smaller than) the typical within-run `std`.
   Large violations mean `std` is under-reporting noise and the
   promotion rule needs tightening before any H2 claim is reportable.
5. **Final test evaluation:** the iter-2 champion's MSE on the locked
   test set is reported **once**, alongside (a) the walk-forward `mean ± std`
   on validation and (b) the pre-COVID sensitivity readout, before the
   project closes. No test-set tuning.

---

## 8. Risks (carried from the charter, with status)

| # | Risk | Mitigation | Status |
|---|---|---|---|
| 1 | Inconsistency between Kaggle and Open-Meteo data | Inner-merge on date; validator surfaces gaps and overlap issues | Mitigated; 0 warnings on current data |
| 2 | Small dataset (~2,100 rows) | Tree-based models prioritized; deep nets out of scope | Mitigated by scope |
| 3 | Weak RRP → demand signal at daily resolution | RRP is optional; an explicit "without RRP" candidate is always present | Mitigated; will revisit in 2-stage iteration |
| 4 | Apparent-temperature feature may add no value | Direct ablation pair in candidate list; H1 will confirm or reject | Open; tested in dry runs |
| 5 | Time leakage | Strict chronological splits; lag features use `.shift` only; test set never touched in loop | Mitigated by construction |
| 6 | Champion config drift across iterations | `splits_summary` recorded in every run report; locked design decisions documented here | Mitigated |
| 7 | Observed-RRP-as-feature semantic leak (this iteration only) | Will be removed in two-stage iteration; flagged in failure-modes table | Acknowledged, deferred |

---

## 9. Workflow

Per iteration:

1. Inspect `experiments/auto_runs/champion.json` and `master_log.csv` (or run `python analyze_runs.py --ablation`).
2. Form a hypothesis ("would adding 14-day lag help?", "is the gain from apparent-temp robust?").
3. Add or modify a candidate in `default_candidates()` so the new candidate differs from the closest existing one by *one* knob.
4. Run `python run_autoresearch.py`.
5. Read the printed summary: did the new candidate beat the champion? By how much vs. the strong baseline?
6. If a new champion was promoted, re-read this file to confirm the locked decisions still hold; if anything would have to change, treat that as a spec amendment.
7. Repeat.

Each iteration's record is the pair: a `run_<id>.json` artifact + the
champion delta printed at the end. The progression across iterations is
the cross-run master log.

---

## 10. Deliverables & milestones

- [x] M0 — Locked decisions documented (this file, README, code).
- [x] M1 — Baseline analysis: validation MSE for both seasonal naives.
- [x] M2 — AutoResearch loop runnable in one command, runtime + budget tracked.
- [x] M3 — Cross-run master log + auto-promoted champion.
- [x] M4 — At least 5 dry-run experiments logged; ablation visible across runs (see §11).
- [x] M5 — At least one non-baseline candidate beats `seasonal_naive_364` by a meaningful margin (iter-1 champion: val MSE 8.88M vs baseline 60.41M, ~85% reduction).
- [x] M6a — Walk-forward CV + bootstrap noise floor + noise-aware champion promotion shipped (`src/split.py::make_walk_forward_folds`, `src/autoresearch.py::_noise_aware_promote`). *(Iter-2 Week 1)*
- [ ] M6b — Two-stage RRP → demand pipeline run end-to-end; H2 verdict (supported / not supported under the 5% effect-size criterion) reported. *(Iter-2 Week 2)*
- [x] M7 — Final test-set evaluation of the champion. Runner is `run_test_evaluation.py` (refits the champion on train+val, predicts on the locked test set, also evaluates both seasonal naives on test for context). Spend the test set deliberately: the script refuses repeat evaluations of the same champion by default. See `FAILURE_ANALYSIS_MEMO.docx` for the recommendation to add a noise floor (L4) before reading the test number as definitive — iter-2 satisfies this via walk-forward `mean ± std`.
- [ ] M8 — Two-stage pipeline tested end-to-end on locked test + pre-COVID sensitivity window. H2 verdict reported with both readouts. Distribution-shift finding (test_minus_pre_covid) reported as the iter-2 closing artifact. *(Iter-2 Week 2)*

---

## 11. Iteration 1 dry-run summary

Five dry runs were executed during scaffolding to populate `master_log.csv`,
exercise the champion-promotion logic, and provide a baseline ablation
story without sklearn. Models used were numpy-only OLS variants
(`src/numpy_models.py::NumpyOLS`) plus the two seasonal naives.

See `REFLECTION.md` for the qualitative retrospective and the actual
master-log table at the end of the dry runs.

---

## 12. Versioning

Bump `Spec version` in the header above whenever any locked decision in §5
changes. All such changes invalidate prior runs and reset
`experiments/auto_runs/champion.json` (delete the file before the next run).
