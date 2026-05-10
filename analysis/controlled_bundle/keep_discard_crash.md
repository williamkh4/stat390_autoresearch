# Controlled Experiments (A/B/C) — Keep / Discard / Crash

Each experiment is sorted into one of three buckets:

- **Keep** — validation MSE beat the strong baseline (`seasonal_naive_364`) by at least 1%. Worth carrying into the next iteration / reporting in a paper.
- **Discard** — fit ran fine but did not beat the strong baseline by a meaningful margin. The configuration is logged but should not be promoted.
- **Crash** — fit/predict raised; the row's error message is the evidence trail.

Strong baseline (`seasonal_naive_364`) val MSE: **60,410,779**.

Counts: **Keep 14** | **Discard 1** | **Crash 0**.

## Keep (14)

| series | label | model | val MSE | test MSE | reason |
|---|---|---|---:|---:|---|
| A | `A1_cal_lag` | mlp | 19,971,784 | 57,972,019 | val MSE 19,971,784 beats strong baseline 60,410,779 by 66.9% |
| A | `A2_cal_temp_lag` | mlp | 10,065,438 | 22,220,155 | val MSE 10,065,438 beats strong baseline 60,410,779 by 83.3% |
| A | `A3_+apparent_temp` | mlp | 9,487,838 | 19,863,512 | val MSE 9,487,838 beats strong baseline 60,410,779 by 84.3% |
| A | `A4_+rolling7` | mlp | 8,879,955 | 18,937,354 | val MSE 8,879,955 beats strong baseline 60,410,779 by 85.3% |
| A | `A5_+RRP_full` | mlp | 10,340,669 | 27,719,174 | val MSE 10,340,669 beats strong baseline 60,410,779 by 82.9% |
| B | `B1_numpy_ols` | numpy_ols | 41,065,610 | 53,218,669 | val MSE 41,065,610 beats strong baseline 60,410,779 by 32.0% |
| B | `B2_ridge` | ridge | 40,971,564 | 53,183,574 | val MSE 40,971,564 beats strong baseline 60,410,779 by 32.2% |
| B | `B3_rf` | rf | 14,373,527 | 27,593,744 | val MSE 14,373,527 beats strong baseline 60,410,779 by 76.2% |
| B | `B4_mlp_champ` | mlp | 8,879,955 | 18,937,354 | val MSE 8,879,955 beats strong baseline 60,410,779 by 85.3% |
| B | `B5_gbm` | gbm | 10,564,234 | 19,593,190 | val MSE 10,564,234 beats strong baseline 60,410,779 by 82.5% |
| C | `C2_lr0.005` | mlp | 10,678,434 | 21,940,066 | val MSE 10,678,434 beats strong baseline 60,410,779 by 82.3% |
| C | `C3_lr0.01` | mlp | 8,879,955 | 18,937,354 | val MSE 8,879,955 beats strong baseline 60,410,779 by 85.3% |
| C | `C4_lr0.02` | mlp | 10,306,467 | 20,919,800 | val MSE 10,306,467 beats strong baseline 60,410,779 by 82.9% |
| C | `C5_lr0.05` | mlp | 8,712,446 | 20,170,309 | val MSE 8,712,446 beats strong baseline 60,410,779 by 85.6% |

## Discard (1)

| series | label | model | val MSE | test MSE | reason |
|---|---|---|---:|---:|---|
| C | `C1_lr0.001` | mlp | 182,452,109 | 228,061,679 | val MSE 182,452,109 does not beat strong baseline 60,410,779 (margin -202.0%) |
