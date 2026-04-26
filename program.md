# Program Specification — Victoria Electricity Demand Forecasting (AutoResearch)

| Field | Value |
|---|---|
| Project | Electricity Grid Demand Forecasting with Weather-Augmented Price Signals |
| Region | Victoria, Australia |
| Time resolution | Daily |
| Spec version | v0.1 (iteration 1, single-stage) |
| Owner | Will Huang |
| Source brief | STAT 390 Week 1 Project Charter |

---

## 1. Research question

> Does incorporating apparent temperature and predicted electricity price (RRP)
> improve the accuracy of daily electricity demand forecasting compared to
> using standard temperature and historical demand alone?

This iteration tests the apparent-temperature half of the question with a
single-stage demand model. The two-stage RRP → demand variant is deferred
to a later iteration; in the current iteration RRP enters as an *observed*
covariate where used, which is acknowledged as optimistic (see §10).

---

## 2. Hypotheses

1. **H1 (apparent temperature):** Including Open-Meteo's apparent-temperature
   features lowers validation MSE compared to the same model with only
   standard temperature features.
2. **H2 (price signal):** Adding RRP as a feature lowers validation MSE.
   Charter flags this as a weak-signal risk; a true test requires the
   two-stage pipeline (deferred).
3. **H3 (non-linearity):** Tree ensembles outperform linear models even
   with the same feature set, because demand-temperature interactions are
   non-linear (e.g., U-shaped: heating below ~12 °C, cooling above ~24 °C).

---

## 3. Scope

**In scope (this iteration):**
- Single-stage daily demand prediction.
- Feature engineering: calendar, temperature, apparent temperature, RRP
  (observed), demand lags 1 and 7, 7-day rolling mean.
- Models: seasonal naive baselines, Ridge, Random Forest, Gradient Boosting,
  numpy OLS variants for ablation.
- AutoResearch loop with cross-run leaderboard and champion tracking.

**Out of scope (this iteration, deferred to later):**
- Two-stage RRP → demand pipeline.
- Sub-daily resolution (intraday, half-hourly).
- Deep learning models.
- Probabilistic forecasts / prediction intervals.
- Final test-set evaluation. The 365-day test set is locked and untouched
  until at least one model demonstrably beats the strong baseline on
  validation by a meaningful margin.

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
| Validation set | 180 days immediately before test | `src/split.py::VAL_DAYS` |
| Train set | All earlier rows | derived |
| Split style | Strict chronological slice | `src/split.py::make_splits` |
| Baselines | `seasonal_naive_7` (weekly floor) and `seasonal_naive_364` (yearly floor; 52-week aligned) | `src/autoresearch.py::baseline_candidates` |
| Reference for new candidates | Current champion in `experiments/auto_runs/champion.json`, auto-promoted on improvement | `src/autoresearch.py::run_loop` |

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

1. **Floor:** at least one non-baseline candidate produces validation MSE
   strictly below the stronger baseline (`seasonal_naive_364`,
   MSE ≈ 60.4 M). A model that doesn't beat trivial yearly recall is not
   adding research value.
2. **H1 acceptance:** A direct ablation pair (same model class, same other
   features) shows that adding apparent-temperature features lowers MSE by
   a margin larger than run-to-run noise (estimated via repeated runs of
   the same candidate).
3. **Final test evaluation** (deferred): the chosen champion's MSE on the
   locked test set is reported once, alongside the validation MSE, before
   the project closes. No test-set tuning.

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
- [ ] M5 — At least one non-baseline candidate beats `seasonal_naive_364` by a meaningful margin.
- [ ] M6 — Two-stage RRP → demand pipeline added as a candidate; H2 tested without the observed-RRP leak.
- [ ] M7 — Final test-set evaluation of the champion. Single number reported, no further tuning.

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
