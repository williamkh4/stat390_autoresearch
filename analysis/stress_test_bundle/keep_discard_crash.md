# MLP/GBM Stress Test (M1-M4, G1-G4) — Keep / Discard / Crash

Each experiment is sorted into one of three buckets:

- **Keep** — validation MSE beat the strong baseline (`seasonal_naive_364`) by at least 1%. Worth carrying into the next iteration / reporting in a paper.
- **Discard** — fit ran fine but did not beat the strong baseline by a meaningful margin. The configuration is logged but should not be promoted.
- **Crash** — fit/predict raised; the row's error message is the evidence trail.

Strong baseline (`seasonal_naive_364`) val MSE: **60,410,779**.

Counts: **Keep 38** | **Discard 2** | **Crash 0**.

## Keep (38)

| series | label | model | val MSE | test MSE | reason |
|---|---|---|---:|---:|---|
| M1 | `M1_cal_lag` | mlp | 19,971,784 | 57,972,019 | val MSE 19,971,784 beats strong baseline 60,410,779 by 66.9% |
| M1 | `M1_cal_temp_lag` | mlp | 10,065,438 | 22,220,155 | val MSE 10,065,438 beats strong baseline 60,410,779 by 83.3% |
| M1 | `M1_cal_temp_apt_lag` | mlp | 9,487,838 | 19,863,512 | val MSE 9,487,838 beats strong baseline 60,410,779 by 84.3% |
| M1 | `M1_champion_features` | mlp | 8,879,955 | 18,937,354 | val MSE 8,879,955 beats strong baseline 60,410,779 by 85.3% |
| M1 | `M1_full_with_RRP` | mlp | 10,340,669 | 27,719,174 | val MSE 10,340,669 beats strong baseline 60,410,779 by 82.9% |
| M2 | `M2_hls32x16` | mlp | 10,974,200 | 22,236,381 | val MSE 10,974,200 beats strong baseline 60,410,779 by 81.8% |
| M2 | `M2_hls64x32` | mlp | 9,085,139 | 19,301,918 | val MSE 9,085,139 beats strong baseline 60,410,779 by 85.0% |
| M2 | `M2_hls128x64` | mlp | 8,879,955 | 18,937,354 | val MSE 8,879,955 beats strong baseline 60,410,779 by 85.3% |
| M2 | `M2_hls256x128x64` | mlp | 8,874,827 | 18,832,466 | val MSE 8,874,827 beats strong baseline 60,410,779 by 85.3% |
| M3 | `M3_alpha1e-05` | mlp | 8,885,243 | 18,979,898 | val MSE 8,885,243 beats strong baseline 60,410,779 by 85.3% |
| M3 | `M3_alpha0.0001` | mlp | 8,879,523 | 18,924,287 | val MSE 8,879,523 beats strong baseline 60,410,779 by 85.3% |
| M3 | `M3_alpha0.001` | mlp | 8,879,955 | 18,937,354 | val MSE 8,879,955 beats strong baseline 60,410,779 by 85.3% |
| M3 | `M3_alpha0.01` | mlp | 8,860,174 | 18,947,632 | val MSE 8,860,174 beats strong baseline 60,410,779 by 85.3% |
| M3 | `M3_alpha0.1` | mlp | 8,846,983 | 19,005,173 | val MSE 8,846,983 beats strong baseline 60,410,779 by 85.4% |
| M4 | `M4_lr0.005` | mlp | 10,678,434 | 21,940,066 | val MSE 10,678,434 beats strong baseline 60,410,779 by 82.3% |
| M4 | `M4_lr0.01` | mlp | 8,879,955 | 18,937,354 | val MSE 8,879,955 beats strong baseline 60,410,779 by 85.3% |
| M4 | `M4_lr0.02` | mlp | 10,306,467 | 20,919,800 | val MSE 10,306,467 beats strong baseline 60,410,779 by 82.9% |
| M4 | `M4_lr0.05` | mlp | 8,712,446 | 20,170,309 | val MSE 8,712,446 beats strong baseline 60,410,779 by 85.6% |
| G1 | `G1_cal_lag` | gbm | 19,901,878 | 56,684,602 | val MSE 19,901,878 beats strong baseline 60,410,779 by 67.1% |
| G1 | `G1_cal_temp_lag` | gbm | 13,450,774 | 26,050,091 | val MSE 13,450,774 beats strong baseline 60,410,779 by 77.7% |
| G1 | `G1_cal_temp_apt_lag` | gbm | 11,375,320 | 21,444,958 | val MSE 11,375,320 beats strong baseline 60,410,779 by 81.2% |
| G1 | `G1_champion_features` | gbm | 10,564,234 | 19,593,190 | val MSE 10,564,234 beats strong baseline 60,410,779 by 82.5% |
| G1 | `G1_full_with_RRP` | gbm | 9,607,938 | 23,305,545 | val MSE 9,607,938 beats strong baseline 60,410,779 by 84.1% |
| G2 | `G2_n100` | gbm | 12,269,499 | 24,503,196 | val MSE 12,269,499 beats strong baseline 60,410,779 by 79.7% |
| G2 | `G2_n200` | gbm | 11,129,793 | 21,051,945 | val MSE 11,129,793 beats strong baseline 60,410,779 by 81.6% |
| G2 | `G2_n300` | gbm | 10,564,234 | 19,593,190 | val MSE 10,564,234 beats strong baseline 60,410,779 by 82.5% |
| G2 | `G2_n500` | gbm | 9,862,589 | 18,251,236 | val MSE 9,862,589 beats strong baseline 60,410,779 by 83.7% |
| G2 | `G2_n1000` | gbm | 9,491,878 | 17,228,501 | val MSE 9,491,878 beats strong baseline 60,410,779 by 84.3% |
| G3 | `G3_depth2` | gbm | 11,997,300 | 20,853,133 | val MSE 11,997,300 beats strong baseline 60,410,779 by 80.1% |
| G3 | `G3_depth3` | gbm | 10,564,234 | 19,593,190 | val MSE 10,564,234 beats strong baseline 60,410,779 by 82.5% |
| G3 | `G3_depth4` | gbm | 10,269,728 | 19,481,865 | val MSE 10,269,728 beats strong baseline 60,410,779 by 83.0% |
| G3 | `G3_depth6` | gbm | 10,061,524 | 19,806,527 | val MSE 10,061,524 beats strong baseline 60,410,779 by 83.3% |
| G3 | `G3_depth8` | gbm | 11,637,722 | 21,170,166 | val MSE 11,637,722 beats strong baseline 60,410,779 by 80.7% |
| G4 | `G4_lr0.01` | gbm | 15,334,772 | 31,594,902 | val MSE 15,334,772 beats strong baseline 60,410,779 by 74.6% |
| G4 | `G4_lr0.03` | gbm | 11,249,259 | 21,223,312 | val MSE 11,249,259 beats strong baseline 60,410,779 by 81.4% |
| G4 | `G4_lr0.05` | gbm | 10,564,234 | 19,593,190 | val MSE 10,564,234 beats strong baseline 60,410,779 by 82.5% |
| G4 | `G4_lr0.1` | gbm | 9,745,509 | 18,415,801 | val MSE 9,745,509 beats strong baseline 60,410,779 by 83.9% |
| G4 | `G4_lr0.2` | gbm | 10,158,919 | 19,073,546 | val MSE 10,158,919 beats strong baseline 60,410,779 by 83.2% |

## Discard (2)

| series | label | model | val MSE | test MSE | reason |
|---|---|---|---:|---:|---|
| M2 | `M2_hls16` | mlp | 2,535,386,499 | 2,635,997,627 | val MSE 2,535,386,499 does not beat strong baseline 60,410,779 (margin -4096.9%) |
| M4 | `M4_lr0.001` | mlp | 182,452,109 | 228,061,679 | val MSE 182,452,109 does not beat strong baseline 60,410,779 (margin -202.0%) |
