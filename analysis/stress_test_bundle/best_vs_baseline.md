# MLP/GBM Stress Test (M1-M4, G1-G4) — Best Result vs. Baseline

The single best experiment in this bundle is **`M4_lr0.05`** (series M4, model `mlp`).

- features: `cal=1 temp=1 apptemp=1 rrp=0 lags=[1, 7] roll=[7]`
- n_features: 24
- runtime: 4.12s

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
| 1 | M4 | `M4_lr0.05` | mlp | 8,712,446 | 20,170,309 | 4.12s |
| 2 | M3 | `M3_alpha0.1` | mlp | 8,846,983 | 19,005,173 | 15.61s |
| 3 | M3 | `M3_alpha0.01` | mlp | 8,860,174 | 18,947,632 | 11.99s |
| 4 | M2 | `M2_hls256x128x64` | mlp | 8,874,827 | 18,832,466 | 13.72s |
| 5 | M3 | `M3_alpha0.0001` | mlp | 8,879,523 | 18,924,287 | 10.08s |
