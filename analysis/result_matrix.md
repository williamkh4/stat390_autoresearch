# Experiment-Result Matrix

Best validation MSE achieved for each (feature_preset, model_type) combination across all AutoResearch runs in `experiments/auto_runs/master_log.csv`. Cells are the **minimum** MSE seen for that pair across all hyperparameter variants tried; `—` means the combination has not been explored yet.

- Runs aggregated: **21**
- Total non-baseline candidates: **104** (84 unique)
- Reference baselines: `seasonal_naive_7` = 141.1M, `seasonal_naive_364` = 60.4M

| feature_preset | numpy_ols | ridge | rf | mlp | gbm |
|---|---:|---:|---:|---:|---:|
| `cal_temp_apptemp_lag1-7_roll7` | — | 41.12M | 13.11M | **8.88M** | 10.62M |
| `full` | 39.87M | 39.97M | 12.27M | 17.43M | 9.10M |
| `cal_temp_apptemp_lag1-7` | 44.26M | 44.19M | 12.50M | 9.49M | 11.23M |
| `cal_temp_lag1-7_roll7` | 41.69M | 41.72M | 13.33M | 9.90M | 11.03M |
| `cal_temp_lag1-7` | — | — | 14.36M | 1377.54M | 12.94M |
| `full_lag1-7-14_roll7-28` | — | — | 17.15M | — | — |
| `cal_temp_apptemp` | — | 92.82M | 25.61M | 44.31M | 19.49M |
| `cal_temp` | — | 99.09M | — | 627.34M | 24.43M |
| `cal_temp_apptemp_lag1-7-14_roll7-28` | — | — | — | 617.97M | — |

**Bold** cell = current overall champion.
