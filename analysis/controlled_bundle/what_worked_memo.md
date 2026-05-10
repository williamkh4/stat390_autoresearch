# Controlled Experiments (A/B/C) — What Actually Worked

A 1-page memo distilled from the per-experiment table. The structure mirrors the sweep design: each series is one knob, so the question for each is *did moving this knob help, and by how much?*

## Headline

- Best of bundle: **`C5_lr0.05`** (series C, model `mlp`).
  - validation MSE: **8,712,446** (test MSE 20,170,309)
  - vs. `seasonal_naive_364` (60,410,779 on val): **+85.6%** improvement.
- Of 15 successful experiments, **14** beat the strong baseline by ≥1%.

## Per-framework summary

| framework | best label | val MSE | test MSE | median val MSE | n experiments |
|---|---|---:|---:|---:|---:|
| `gbm` | `B5_gbm` | 10,564,234 | 19,593,190 | 10,564,234 | 1 |
| `mlp` | `C5_lr0.05` | 8,712,446 | 20,170,309 | 10,065,438 | 11 |
| `numpy_ols` | `B1_numpy_ols` | 41,065,610 | 53,218,669 | 41,065,610 | 1 |
| `rf` | `B3_rf` | 14,373,527 | 27,593,744 | 14,373,527 | 1 |
| `ridge` | `B2_ridge` | 40,971,564 | 53,183,574 | 40,971,564 | 1 |

## Per-series read-out

### Series A (5 experiments, 5 successful, 0 crashes)

- best: `A4_+rolling7` (val MSE 8,879,955, test MSE 18,937,354)
- worst: `A1_cal_lag` (val MSE 19,971,784)
- spread within series: **55.5%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `A1_cal_lag` | 19,971,784 | 57,972,019 | 4,469 | 7,614 | 3,464 | 5,679 |
| `A2_cal_temp_lag` | 10,065,438 | 22,220,155 | 3,173 | 4,714 | 2,422 | 3,665 |
| `A3_+apparent_temp` | 9,487,838 | 19,863,512 | 3,080 | 4,457 | 2,396 | 3,542 |
| `A4_+rolling7` | 8,879,955 | 18,937,354 | 2,980 | 4,352 | 2,320 | 3,468 |
| `A5_+RRP_full` | 10,340,669 | 27,719,174 | 3,216 | 5,265 | 2,501 | 4,274 |

### Series B (5 experiments, 5 successful, 0 crashes)

- best: `B4_mlp_champ` (val MSE 8,879,955, test MSE 18,937,354)
- worst: `B1_numpy_ols` (val MSE 41,065,610)
- spread within series: **78.4%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `B1_numpy_ols` | 41,065,610 | 53,218,669 | 6,408 | 7,295 | 5,082 | 5,749 |
| `B2_ridge` | 40,971,564 | 53,183,574 | 6,401 | 7,293 | 5,073 | 5,749 |
| `B3_rf` | 14,373,527 | 27,593,744 | 3,791 | 5,253 | 2,967 | 4,173 |
| `B4_mlp_champ` | 8,879,955 | 18,937,354 | 2,980 | 4,352 | 2,320 | 3,468 |
| `B5_gbm` | 10,564,234 | 19,593,190 | 3,250 | 4,426 | 2,453 | 3,535 |

### Series C (5 experiments, 5 successful, 0 crashes)

- best: `C5_lr0.05` (val MSE 8,712,446, test MSE 20,170,309)
- worst: `C1_lr0.001` (val MSE 182,452,109)
- spread within series: **95.2%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `C1_lr0.001` | 182,452,109 | 228,061,679 | 13,507 | 15,102 | 10,911 | 11,976 |
| `C2_lr0.005` | 10,678,434 | 21,940,066 | 3,268 | 4,684 | 2,478 | 3,700 |
| `C3_lr0.01` | 8,879,955 | 18,937,354 | 2,980 | 4,352 | 2,320 | 3,468 |
| `C4_lr0.02` | 10,306,467 | 20,919,800 | 3,210 | 4,574 | 2,473 | 3,656 |
| `C5_lr0.05` | 8,712,446 | 20,170,309 | 2,952 | 4,491 | 2,212 | 3,576 |

## How to read this

- Within a series, **everything except one knob is held fixed**, so a spread tells you how sensitive the metric is to that knob alone.
- The validation column is the optimisation surface; the test column is reported for honesty. A series where val improves but test doesn't (or moves the wrong way) is a generalisation warning, not a win.
- Beating the strong baseline (`seasonal_naive_364`, yearly recall) is the floor for *adding research value*; anything weaker means weather + lag features didn't pay off for that configuration.

## Caveats

- The framework is deterministic, so a single number per experiment is a point estimate without a noise floor. Treat differences smaller than ~5% with skepticism. (See `FAILURE_ANALYSIS_MEMO.docx` for the recommendation to add a bootstrap noise estimate before reading any single number as definitive.)
- Test scores here come from a **train-only** fit predicting on the locked test window — same protocol as the auto loop's `--evaluate-on-test`. The deployment-style refit on train+val is in `run_test_evaluation.py`.
- The auto loop's champion promotion is decided on validation MSE only; test numbers in this bundle are diagnostic, not selectorial.
