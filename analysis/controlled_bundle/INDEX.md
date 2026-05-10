# Controlled Experiments (A/B/C) — Experiment Log Bundle

Self-contained analysis package for one controlled experiment set. Read this file first; the per-deliverable files below are the answers to specific questions about the run.

## Source

- Results CSV: `experiments/controlled/controlled_results.csv`
- Total experiments: **15**
- Series: **3** (A=5, B=5, C=5)
- Crashes: **0**
- Baselines included: `seasonal_naive_7`, `seasonal_naive_364`

## Deliverables

| # | File | What it answers |
|---|---|---|
| 1 | `INDEX.md` (this file) | What's in this bundle, where each artifact lives |
| 2 | `metric_trajectory.png` | How val + test RMSE/MAE move across the sweep, vs. baselines |
| 3 | `keep_discard_crash.md` | For every experiment: keep / discard / crash, with reason |
| 4 | `best_vs_baseline.md` | The single best experiment side-by-side with the baselines |
| 5 | `what_worked_memo.md` | Per-series narrative: which knob moved the needle, and which didn't |

## Per-experiment summary

| series | label | model_type | n_features | mse_demand | rmse_demand | mae_demand | mse_demand_test | rmse_demand_test | mae_demand_test | runtime_sec | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | seasonal_naive_7 | baseline | 0 | 141,138,184 | 11,880 | 9,762 | 207,834,241 | 14,416 | 12,113 | 0.01 |  |
| baseline | seasonal_naive_364 | baseline | 0 | 60,410,779 | 7,772 | 6,271 | 224,849,544 | 14,995 | 11,031 | 0.01 |  |
| A | A1_cal_lag | mlp | 10 | 19,971,784 | 4,469 | 3,464 | 57,972,019 | 7,614 | 5,679 | 9.46 |  |
| A | A2_cal_temp_lag | mlp | 20 | 10,065,438 | 3,173 | 2,422 | 22,220,155 | 4,714 | 3,665 | 6.33 |  |
| A | A3_+apparent_temp | mlp | 23 | 9,487,838 | 3,080 | 2,396 | 19,863,512 | 4,457 | 3,542 | 10.36 |  |
| A | A4_+rolling7 | mlp | 24 | 8,879,955 | 2,980 | 2,320 | 18,937,354 | 4,352 | 3,468 | 8.21 |  |
| A | A5_+RRP_full | mlp | 25 | 10,340,669 | 3,216 | 2,501 | 27,719,174 | 5,265 | 4,274 | 6.48 |  |
| B | B1_numpy_ols | numpy_ols | 24 | 41,065,610 | 6,408 | 5,082 | 53,218,669 | 7,295 | 5,749 | 0.00 |  |
| B | B2_ridge | ridge | 24 | 40,971,564 | 6,401 | 5,073 | 53,183,574 | 7,293 | 5,749 | 2.51 |  |
| B | B3_rf | rf | 24 | 14,373,527 | 3,791 | 2,967 | 27,593,744 | 5,253 | 4,173 | 3.44 |  |
| B | B4_mlp_champ | mlp | 24 | 8,879,955 | 2,980 | 2,320 | 18,937,354 | 4,352 | 3,468 | 4.42 |  |
| B | B5_gbm | gbm | 24 | 10,564,234 | 3,250 | 2,453 | 19,593,190 | 4,426 | 3,535 | 1.60 |  |
| C | C1_lr0.001 | mlp | 24 | 182,452,109 | 13,507 | 10,911 | 228,061,679 | 15,102 | 11,976 | 4.57 |  |
| C | C2_lr0.005 | mlp | 24 | 10,678,434 | 3,268 | 2,478 | 21,940,066 | 4,684 | 3,700 | 4.35 |  |
| C | C3_lr0.01 | mlp | 24 | 8,879,955 | 2,980 | 2,320 | 18,937,354 | 4,352 | 3,468 | 4.74 |  |
| C | C4_lr0.02 | mlp | 24 | 10,306,467 | 3,210 | 2,473 | 20,919,800 | 4,574 | 3,656 | 3.23 |  |
| C | C5_lr0.05 | mlp | 24 | 8,712,446 | 2,952 | 2,212 | 20,170,309 | 4,491 | 3,576 | 3.13 |  |
