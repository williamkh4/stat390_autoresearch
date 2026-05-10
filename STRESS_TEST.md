# MLP / GBM Stress Test

A targeted set of controlled sweeps that drills into the two
best-performing model frameworks in the project (MLP and GBM) and ablates
them along the **levers that matter inside each framework's own context**:
features and the most important hyperparameters.

Where `EXPERIMENTS.md` describes the project-wide A/B/C controlled set
(features × model class × MLP learning rate around the *current
champion*), this stress test treats MLP and GBM as separate research
artifacts and asks "given that this is the framework we're betting on,
what knob actually moves the metric?"

The runner is `run_stress_test.py`; artifacts land in
`experiments/stress_test/` and the analysis bundle lives at
`analysis/stress_test_bundle/`.

---

## Pivots (held constant within each framework's sweeps)

| pivot | value |
|---|---|
| feature preset (default) | `cal + temp + apparent_temp + lag[1, 7] + roll[7]` (24 features) |
| MLP base spec | `hidden_layer_sizes=(128,64)`, `activation='relu'`, `solver='adam'`, `alpha=1e-3`, `learning_rate_init=0.01`, `max_iter=500`, `random_state=0` |
| GBM base spec | `n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `subsample=1.0`, `random_state=0` |

The MLP base spec is the current champion's exact configuration. The GBM
base spec is the strongest GBM that has surfaced in the auto-loop search
space, used here as a sensible centre to sweep around.

---

## MLP series (M1–M4)

Each series varies one knob; everything else is held at the MLP base spec
and the default feature preset.

### M1 — Feature selection

**Asks:** which feature group buys the gain in the MLP framework?

| label | feature config |
|---|---|
| M1 `cal_lag` | calendar + demand lag 1, 7 |
| M1 `cal_temp_lag` | + Kaggle/Open-Meteo temperature group |
| M1 `cal_temp_apt_lag` | + apparent temperature (H1) |
| M1 `champion_features` ★ | + 7-day rolling demand mean (champion) |
| M1 `full_with_RRP` | + RRP (observed; semantic leak — see `program.md` §10) |

### M2 — Architecture (`hidden_layer_sizes`)

**Asks:** is the MLP at the right capacity, or under/over-parameterised?

| label | hidden_layer_sizes |
|---|---|
| M2 `hls16` | (16,) |
| M2 `hls32x16` | (32, 16) |
| M2 `hls64x32` | (64, 32) |
| M2 `hls128x64` ★ | (128, 64) (champion arch) |
| M2 `hls256x128x64` | (256, 128, 64) |

### M3 — L2 regularisation (`alpha`)

**Asks:** is the model under- or over-regularised? `alpha=1e-3` is the
champion; we sweep two orders of magnitude on either side.

| label | alpha |
|---|---|
| M3 `alpha1e-05` | 1e-5 |
| M3 `alpha0.0001` | 1e-4 |
| M3 `alpha0.001` ★ | 1e-3 (champion) |
| M3 `alpha0.01` | 1e-2 |
| M3 `alpha0.1` | 1e-1 |

### M4 — `learning_rate_init`

**Asks:** is the MLP optimiser at a reasonable step size? Mirrors Series
C in the controlled bundle on purpose, so all four MLP knobs sit in one
comparable plot.

| label | learning_rate_init |
|---|---|
| M4 `lr0.001` | 0.001 |
| M4 `lr0.005` | 0.005 |
| M4 `lr0.01` ★ | 0.01 (champion) |
| M4 `lr0.02` | 0.02 |
| M4 `lr0.05` | 0.05 |

---

## GBM series (G1–G4)

Each series varies one knob; everything else is held at the GBM base spec
and the default feature preset.

### G1 — Feature selection

**Asks:** does GBM benefit from the same feature set MLP does, or is the
tree ensemble agnostic to apparent temperature?

| label | feature config |
|---|---|
| G1 `cal_lag` | calendar + demand lag 1, 7 |
| G1 `cal_temp_lag` | + temperature group |
| G1 `cal_temp_apt_lag` | + apparent temperature |
| G1 `champion_features` ★ | + roll7 (champion features) |
| G1 `full_with_RRP` | + RRP |

### G2 — `n_estimators`

**Asks:** is 300 trees enough, or are we under-bagging?

| label | n_estimators |
|---|---|
| G2 `n100` | 100 |
| G2 `n200` | 200 |
| G2 `n300` ★ | 300 (default) |
| G2 `n500` | 500 |
| G2 `n1000` | 1000 |

### G3 — `max_depth`

**Asks:** are individual trees the right depth for the demand-temperature
interaction (likely shallow + non-linear)?

| label | max_depth |
|---|---|
| G3 `depth2` | 2 |
| G3 `depth3` ★ | 3 (default) |
| G3 `depth4` | 4 |
| G3 `depth6` | 6 |
| G3 `depth8` | 8 |

### G4 — `learning_rate` (shrinkage)

**Asks:** is the shrinkage well-matched to the tree count?

| label | learning_rate |
|---|---|
| G4 `lr0.01` | 0.01 |
| G4 `lr0.03` | 0.03 |
| G4 `lr0.05` ★ | 0.05 (default) |
| G4 `lr0.1` | 0.1 |
| G4 `lr0.2` | 0.2 |

---

## How to run

```bash
python run_stress_test.py
python build_experiment_bundle.py \
    --results-csv experiments/stress_test/stress_results.csv \
    --out-dir analysis/stress_test_bundle \
    --title "MLP/GBM Stress Test (M1-M4, G1-G4)"
```

The first command writes the raw experiment log
(`experiments/stress_test/stress_results.csv` + per-series JSONs +
`stress_master.csv` in master_log shape). The second produces the
five-deliverable bundle from those results.

---

## How to read the bundle

The bundle's five files answer different questions:

| File | Reads like… |
|---|---|
| `INDEX.md` | a table of contents — what's here and what each artifact answers |
| `metric_trajectory.png` | a chart — RMSE/MAE per experiment, val + test, with baselines |
| `keep_discard_crash.md` | a triage list — which configs to carry forward |
| `best_vs_baseline.md` | a one-screen verdict on how the best fit lined up vs baselines |
| `what_worked_memo.md` | a research memo — per-series narrative + per-framework summary |

Read `INDEX.md` first if you're new to the bundle, then `what_worked_memo.md`
for the narrative, and use the other three when you want to verify a
specific claim against the underlying numbers.

---

## What this stress test is *not*

- It does **not** rerun the full auto-loop search space; it sweeps four
  knobs per framework, holding everything else fixed.
- It does **not** reset the project champion — the auto-loop's champion
  promotion is unchanged.
- It does **not** repeat-fit each candidate (the framework is
  deterministic by design); a single number per cell is a point estimate.
  See `FAILURE_ANALYSIS_MEMO.docx` for the standing recommendation to
  layer a bootstrap noise floor before reading any of these numbers as
  statistically definitive.
