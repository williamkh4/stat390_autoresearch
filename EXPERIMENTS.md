# Controlled Experiments

A documented, deliberate sweep where **exactly one variable changes per
run** and everything else is held fixed. This complements the auto-driving
loop: that loop is great at *exploring* the space, but its choices are
stochastic, so when we want to *interpret* a result (e.g. "did adding
apparent_temp help, or was it just noise?") we run a controlled set
instead.

The controlled set lives in `run_controlled_experiments.py` and writes
artifacts to `experiments/controlled/`, separate from the auto-driving
master log so it doesn't pollute history-based dedup or champion
promotion.

---

## Pivot point

All three series ablate around the **current AutoResearch champion**
(per `experiments/auto_runs/champion.json`):

| field | value |
|---|---|
| candidate name | `mlp__cal_temp_apptemp_lag1-7_roll7__hls128x64_actrelu_solveradam_alpha0.001_lrinit0.01_iter500` |
| validation MSE | 8,879,955 |
| model class | MLP (128, 64), ReLU, adam, alpha=0.001, lr_init=0.01, max_iter=500 |
| feature config | calendar + temp + apparent_temp + demand_lag[1, 7] + rolling[7] |
| n_features | 24 |
| set in run | `8d36999d` |

If the champion changes, update `CHAMPION_FEATURE_CONFIG` and
`CHAMPION_MLP_KWARGS` constants at the top of
`run_controlled_experiments.py` and re-run.

---

## Series A — Feature ablation

**Held fixed:** model = champion's MLP spec (architecture + hyperparams).
**Varied:** which feature group is included.
**Asks:** which feature toggles produced the champion's gain?

| label | feature config | one-knob delta from previous |
|---|---|---|
| A1 | `cal + lag[1, 7]` | (entry point) |
| A2 | `cal + temp + lag[1, 7]` | added `temp` |
| A3 | `cal + temp + apptemp + lag[1, 7]` | added `apparent_temp` (H1) |
| A4 ★ | `cal + temp + apptemp + lag[1, 7] + roll[7]` | added `roll7` (champion) |
| A5 | `cal + temp + apptemp + lag[1, 7] + roll[7] + RRP` | added observed RRP |

A4 is the champion's exact feature config. Comparing A2→A3 isolates the
apparent-temperature contribution under the champion's MLP — this is the
H1 confirmation we couldn't get from unregularized OLS.

---

## Series B — Model class ablation

**Held fixed:** features = champion's preset
(`cal + temp + apptemp + lag[1, 7] + roll[7]`).
**Varied:** model class (and minimal sensible hyperparameters per class).
**Asks:** how much of the champion's gain comes from the model choice
versus the features?

| label | model spec |
|---|---|
| B1 | `numpy_ols` (alpha=1.0) |
| B2 | `ridge` (alpha=1.0) |
| B3 | `rf` (n=200, max_depth=None, leaf=5, max_features='sqrt') |
| B4 ★ | `mlp` champion (hls=(128,64), relu, adam, alpha=0.001, lr=0.01) |
| B5 | `gbm` (n=300, max_depth=3, lr=0.05, subsample=1.0) |

Cross-tab readout once you have run B: the model-class column of the
result matrix becomes a single ablation slice through the champion's row.

---

## Series C — MLP learning-rate sweep

**Held fixed:** features = champion's preset, model = champion's MLP
architecture (everything except `learning_rate_init`).
**Varied:** `learning_rate_init` ∈ {0.001, 0.005, 0.01, 0.02, 0.05}.
**Asks:** is the champion's lr=0.01 a peak, an edge, or a plateau?

| label | learning_rate_init |
|---|---|
| C1 | 0.001 |
| C2 | 0.005 |
| C3 ★ | 0.01 (champion) |
| C4 | 0.02 |
| C5 | 0.05 |

If the surface is convex and lr=0.01 is the peak, both C2/C4 should
underperform C3 by similar amounts. If the surface is monotone, the
champion is on an edge and the next iteration should explore further in
the direction of the gradient.

---

## How to run

```bash
python run_controlled_experiments.py
```

Output:
```
experiments/controlled/controlled_results.csv     # one row per experiment
experiments/controlled/controlled_master.csv      # master_log shape, reusable
experiments/controlled/series_A.json              # per-series detail
experiments/controlled/series_B.json
experiments/controlled/series_C.json
```

---

## As-run cross-tab

For the auto-driving runs (not the controlled set yet), the matrix of
best validation MSE across (feature_preset × model_type) lives at
`analysis/result_matrix.md` and `analysis/result_matrix.csv`. That table
is the at-a-glance answer to "which combinations have I tried, and how
do they compare?". Run it through `analyze_runs.py --ablation` if you
want per-candidate mean/std (note: std is currently 0.0 because the
framework is deterministic — see `FAILURE_ANALYSIS_MEMO.md` for why
that's a problem and what to do about it).

The metric-over-time plot (`analysis/metric_over_time.png`) shows the
champion-after-run trajectory and per-run candidate scatter; use it
together with the controlled-experiment series to separate "trend
across runs" from "ablation within a run."
