# MLP/GBM Stress Test (M1-M4, G1-G4) — What Actually Worked

A 1-page memo distilled from the per-experiment table. The structure mirrors the sweep design: each series is one knob, so the question for each is *did moving this knob help, and by how much?*

## Headline

- Best of bundle: **`M4_lr0.05`** (series M4, model `mlp`).
  - validation MSE: **8,712,446** (test MSE 20,170,309)
  - vs. `seasonal_naive_364` (60,410,779 on val): **+85.6%** improvement.
- Of 40 successful experiments, **38** beat the strong baseline by ≥1%.

## Per-framework summary

| framework | best label | val MSE | test MSE | median val MSE | n experiments |
|---|---|---:|---:|---:|---:|
| `gbm` | `G2_n1000` | 9,491,878 | 17,228,501 | 10,564,234 | 20 |
| `mlp` | `M4_lr0.05` | 8,712,446 | 20,170,309 | 8,985,191 | 20 |

## Per-series read-out

### Series G1 (5 experiments, 5 successful, 0 crashes)

- best: `G1_full_with_RRP` (val MSE 9,607,938, test MSE 23,305,545)
- worst: `G1_cal_lag` (val MSE 19,901,878)
- spread within series: **51.7%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `G1_cal_lag` | 19,901,878 | 56,684,602 | 4,461 | 7,529 | 3,469 | 5,512 |
| `G1_cal_temp_lag` | 13,450,774 | 26,050,091 | 3,668 | 5,104 | 2,721 | 4,067 |
| `G1_cal_temp_apt_lag` | 11,375,320 | 21,444,958 | 3,373 | 4,631 | 2,611 | 3,720 |
| `G1_champion_features` | 10,564,234 | 19,593,190 | 3,250 | 4,426 | 2,453 | 3,535 |
| `G1_full_with_RRP` | 9,607,938 | 23,305,545 | 3,100 | 4,828 | 2,323 | 3,884 |

### Series G2 (5 experiments, 5 successful, 0 crashes)

- best: `G2_n1000` (val MSE 9,491,878, test MSE 17,228,501)
- worst: `G2_n100` (val MSE 12,269,499)
- spread within series: **22.6%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `G2_n100` | 12,269,499 | 24,503,196 | 3,503 | 4,950 | 2,654 | 3,879 |
| `G2_n200` | 11,129,793 | 21,051,945 | 3,336 | 4,588 | 2,519 | 3,646 |
| `G2_n300` | 10,564,234 | 19,593,190 | 3,250 | 4,426 | 2,453 | 3,535 |
| `G2_n500` | 9,862,589 | 18,251,236 | 3,140 | 4,272 | 2,363 | 3,423 |
| `G2_n1000` | 9,491,878 | 17,228,501 | 3,081 | 4,151 | 2,308 | 3,315 |

### Series G3 (5 experiments, 5 successful, 0 crashes)

- best: `G3_depth6` (val MSE 10,061,524, test MSE 19,806,527)
- worst: `G3_depth2` (val MSE 11,997,300)
- spread within series: **16.1%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `G3_depth2` | 11,997,300 | 20,853,133 | 3,464 | 4,567 | 2,714 | 3,631 |
| `G3_depth3` | 10,564,234 | 19,593,190 | 3,250 | 4,426 | 2,453 | 3,535 |
| `G3_depth4` | 10,269,728 | 19,481,865 | 3,205 | 4,414 | 2,443 | 3,508 |
| `G3_depth6` | 10,061,524 | 19,806,527 | 3,172 | 4,450 | 2,435 | 3,524 |
| `G3_depth8` | 11,637,722 | 21,170,166 | 3,411 | 4,601 | 2,717 | 3,632 |

### Series G4 (5 experiments, 5 successful, 0 crashes)

- best: `G4_lr0.1` (val MSE 9,745,509, test MSE 18,415,801)
- worst: `G4_lr0.01` (val MSE 15,334,772)
- spread within series: **36.4%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `G4_lr0.01` | 15,334,772 | 31,594,902 | 3,916 | 5,621 | 2,958 | 4,417 |
| `G4_lr0.03` | 11,249,259 | 21,223,312 | 3,354 | 4,607 | 2,573 | 3,658 |
| `G4_lr0.05` | 10,564,234 | 19,593,190 | 3,250 | 4,426 | 2,453 | 3,535 |
| `G4_lr0.1` | 9,745,509 | 18,415,801 | 3,122 | 4,291 | 2,375 | 3,455 |
| `G4_lr0.2` | 10,158,919 | 19,073,546 | 3,187 | 4,367 | 2,522 | 3,451 |

### Series M1 (5 experiments, 5 successful, 0 crashes)

- best: `M1_champion_features` (val MSE 8,879,955, test MSE 18,937,354)
- worst: `M1_cal_lag` (val MSE 19,971,784)
- spread within series: **55.5%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `M1_cal_lag` | 19,971,784 | 57,972,019 | 4,469 | 7,614 | 3,464 | 5,679 |
| `M1_cal_temp_lag` | 10,065,438 | 22,220,155 | 3,173 | 4,714 | 2,422 | 3,665 |
| `M1_cal_temp_apt_lag` | 9,487,838 | 19,863,512 | 3,080 | 4,457 | 2,396 | 3,542 |
| `M1_champion_features` | 8,879,955 | 18,937,354 | 2,980 | 4,352 | 2,320 | 3,468 |
| `M1_full_with_RRP` | 10,340,669 | 27,719,174 | 3,216 | 5,265 | 2,501 | 4,274 |

### Series M2 (5 experiments, 5 successful, 0 crashes)

- best: `M2_hls256x128x64` (val MSE 8,874,827, test MSE 18,832,466)
- worst: `M2_hls16` (val MSE 2,535,386,499)
- spread within series: **99.6%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `M2_hls16` | 2,535,386,499 | 2,635,997,627 | 50,353 | 51,342 | 41,913 | 44,311 |
| `M2_hls32x16` | 10,974,200 | 22,236,381 | 3,313 | 4,716 | 2,557 | 3,694 |
| `M2_hls64x32` | 9,085,139 | 19,301,918 | 3,014 | 4,393 | 2,284 | 3,452 |
| `M2_hls128x64` | 8,879,955 | 18,937,354 | 2,980 | 4,352 | 2,320 | 3,468 |
| `M2_hls256x128x64` | 8,874,827 | 18,832,466 | 2,979 | 4,340 | 2,299 | 3,444 |

### Series M3 (5 experiments, 5 successful, 0 crashes)

- best: `M3_alpha0.1` (val MSE 8,846,983, test MSE 19,005,173)
- worst: `M3_alpha1e-05` (val MSE 8,885,243)
- spread within series: **0.4%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `M3_alpha1e-05` | 8,885,243 | 18,979,898 | 2,981 | 4,357 | 2,326 | 3,470 |
| `M3_alpha0.0001` | 8,879,523 | 18,924,287 | 2,980 | 4,350 | 2,324 | 3,469 |
| `M3_alpha0.001` | 8,879,955 | 18,937,354 | 2,980 | 4,352 | 2,320 | 3,468 |
| `M3_alpha0.01` | 8,860,174 | 18,947,632 | 2,977 | 4,353 | 2,319 | 3,469 |
| `M3_alpha0.1` | 8,846,983 | 19,005,173 | 2,974 | 4,359 | 2,320 | 3,474 |

### Series M4 (5 experiments, 5 successful, 0 crashes)

- best: `M4_lr0.05` (val MSE 8,712,446, test MSE 20,170,309)
- worst: `M4_lr0.001` (val MSE 182,452,109)
- spread within series: **95.2%** of the worst value — the bigger this number, the more sensitive the metric is to *this* knob.

| label | val MSE | test MSE | val RMSE | test RMSE | val MAE | test MAE |
|---|---:|---:|---:|---:|---:|---:|
| `M4_lr0.001` | 182,452,109 | 228,061,679 | 13,507 | 15,102 | 10,911 | 11,976 |
| `M4_lr0.005` | 10,678,434 | 21,940,066 | 3,268 | 4,684 | 2,478 | 3,700 |
| `M4_lr0.01` | 8,879,955 | 18,937,354 | 2,980 | 4,352 | 2,320 | 3,468 |
| `M4_lr0.02` | 10,306,467 | 20,919,800 | 3,210 | 4,574 | 2,473 | 3,656 |
| `M4_lr0.05` | 8,712,446 | 20,170,309 | 2,952 | 4,491 | 2,212 | 3,576 |

## How to read this

- Within a series, **everything except one knob is held fixed**, so a spread tells you how sensitive the metric is to that knob alone.
- The validation column is the optimisation surface; the test column is reported for honesty. A series where val improves but test doesn't (or moves the wrong way) is a generalisation warning, not a win.
- Beating the strong baseline (`seasonal_naive_364`, yearly recall) is the floor for *adding research value*; anything weaker means weather + lag features didn't pay off for that configuration.

## Caveats

- The framework is deterministic, so a single number per experiment is a point estimate without a noise floor. Treat differences smaller than ~5% with skepticism. (See `FAILURE_ANALYSIS_MEMO.docx` for the recommendation to add a bootstrap noise estimate before reading any single number as definitive.)
- Test scores here come from a **train-only** fit predicting on the locked test window — same protocol as the auto loop's `--evaluate-on-test`. The deployment-style refit on train+val is in `run_test_evaluation.py`.
- The auto loop's champion promotion is decided on validation MSE only; test numbers in this bundle are diagnostic, not selectorial.
