# Controlled Experiments (A/B/C) — Best Result vs. Baseline

The single best experiment in this bundle is **`C5_lr0.05`** (series C, model `mlp`).

- features: `cal=1 temp=1 apptemp=1 rrp=0 lags=[1, 7] roll=[7]`
- n_features: 24
- runtime: 3.13s

## Side-by-side

| metric | best (val) | best (test) | `seasonal_naive_7` (val) | `seasonal_naive_7` (test) | `seasonal_naive_364` (val) | `seasonal_naive_364` (test) | Δ vs. strong (val) | Δ vs. strong (test) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE | 8,712,446 | 20,170,309 | 141,138,184 | 207,834,241 | 60,410,779 | 224,849,544 | +85.6% | +91.0% |
| RMSE | 2,952 | 4,491 | 11,880 | 14,416 | 7,772 | 14,995 | +62.0% | +70.0% |
| MAE | 2,212 | 3,576 | 9,762 | 12,113 | 6,271 | 11,031 | +64.7% | +67.6% |

Δ is `(baseline − best) / baseline`, so positive = best is better than baseline.

## Top 5 by validation MSE

| rank | series | label | model | val MSE | test MSE | runtime |
|---:|---|---|---|---:|---:|---:|
| 1 | C | `C5_lr0.05` | mlp | 8,712,446 | 20,170,309 | 3.13s |
| 2 | A | `A4_+rolling7` | mlp | 8,879,955 | 18,937,354 | 8.21s |
| 3 | B | `B4_mlp_champ` | mlp | 8,879,955 | 18,937,354 | 4.42s |
| 4 | C | `C3_lr0.01` | mlp | 8,879,955 | 18,937,354 | 4.74s |
| 5 | A | `A3_+apparent_temp` | mlp | 9,487,838 | 19,863,512 | 10.36s |
