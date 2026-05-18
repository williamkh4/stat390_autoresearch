# Iter-2 Scope-Narrowing Ablation Tables

This file documents *why* iteration 2 narrowed scope the way it did.
Every cell below is justified by a number already in
`analysis/result_matrix.{csv,md}` or by an entry in `ERROR_TAXONOMY.md`.

Read this file together with `program.md` §1 (the v0.2 research
statement) and `TWO_WEEK_PLAN.md` (the build calendar). Together they
form the iter-2 scope lock.

---

## Shape 1 — Hypothesis-Evidence Matrix

How iteration 1's evidence shaped iteration 2's research focus.

| Hypothesis | Iter-1 evidence | Iter-2 disposition |
|---|---|---|
| **H1: apparent_temp helps** | Champion uses apparent_temp; the controlled-experiments bundle's A2 → A3 pair shows the gain is real (val MSE 10.07M → 9.49M, ~5.7% reduction at the same MLP spec); the H3 result-matrix shows apparent_temp moves the best-cell across model classes. | **Frozen as locked feature; not re-ablated.** |
| **H2: RRP helps** | Untested — `full` candidates used **observed** RRP at forecast time (L1 leak); the current champion happens to drop RRP entirely. | **Primary iter-2 work: two-stage pipeline (stage-1 RRP predictor → stage-2 demand model) implemented in `src/predict_rrp.py`.** Minimum effect size **5% relative MSE reduction**, gated by the noise-aware promotion rule. |
| **H3: non-linear models win** | Trees and MLP 4× better than linear across every preset (`result_matrix`); best MLP 8.88M vs best Ridge 41.12M. | **Confirmed; drop linear from active search.** One numpy_ols + one ridge entry kept as sanity rows only (`ModelSpec.is_sanity_only=True`). |
| **H4 (new): val→test shift is data, not model** | `seasonal_naive_364` val MSE 60.41M → test MSE 188.09M (3.11×), deterministic for both baselines. Champion val→test ratio 8.88M → 18.94M (2.13×) — better, but the absolute gap is still dominated by the test-window distribution. | **Add 365-day pre-COVID sensitivity readout (one-shot alongside locked test) via `run_test_evaluation.py --pre-covid-sensitivity`.** Disentangles "model quality" from "COVID regime change." |
| **M1 (new methodology): bootstrap noise floor is calibrated** | Iter-1 reported deterministic std=0 because every comparison was a single 180-day hold-out. No validation that any leaderboard gap was real. | **Walk-forward CV with `mean ± std`; noise-aware champion promotion: `mean_chal + std_chal < mean_champ − std_champ`.** Calibrated by replaying the iter-1 champion under walk-forward (Week 1, Day 2 of the plan). |

---

## Shape 2 — Search-Space Narrowing

What changed dimensionally between iter-1 and iter-2, and why.

| Dimension | Iter-1 scope | Iter-2 scope | Justification (with numbers) |
|---|---|---|---|
| **Model classes** | 5 (numpy_ols, ridge, rf, mlp, gbm) | 3 active (rf, mlp, gbm) + 1 numpy_ols + 1 ridge as `is_sanity_only=True` smoke rows | Linear corner produces no improvement (`Signal Failure S-class` in `ERROR_TAXONOMY.md`); best Ridge 41.12M ≫ best MLP 8.88M |
| **Feature presets** | 13 in `FEATURE_PRESETS` | 5 "primary" + 3 predicted-RRP variants + 5 legacy non-primary (deprioritised) | Five presets had at least one non-linear cell <15M val MSE in iter-1's result matrix; the others never produced a competitive cell |
| **RRP usage** | observed only (leaky) | predicted (two-stage, `FeatureConfig.use_predicted_rrp`); observed deprecated and `master_log.uses_observed_rrp=True` for excludable history rows | L1 in `ERROR_TAXONOMY.md` |
| **Validation surface** | single fixed 180-day window (`make_splits`) | walk-forward CV: `val_size=180`, `step=90`, `min_train=730`, **10 folds, expanding train** | L4 (noise floor) + structural answer to "is val_180 representative?" |
| **Champion promotion** | strict `lower-MSE-wins` | noise-aware: `mean_challenger + std_challenger < mean_champion − std_champion` | L4 — iter-1 promoted on differences smaller than the noise floor |
| **Test usage** | locked, untouched | one-shot at end + one-shot pre-COVID sensitivity (same script, separate window) | L5 (one-shot runner shipped) + L6 (distribution shift visible in baselines) |
| **MLP variants** | all 10 in `MODEL_SPECS` | exclude 3 that produced >500M val MSE in iter-1 (`known_bad=True`); auto-generator skips them | C5 (catastrophic non-convergence) — iter-1 result matrix shows these cells |
| **`--promote-on=test` flag** | new design surface, mostly diagnostic | retained but never used in iter-2 by default (escape hatch only) | A2 (methodology trapdoor); use only with explicit advisor sign-off |
| **Stage-1 RRP model** | n/a | GBM-300/3/0.05 default (`src/predict_rrp.py`); no model-class sweep | Iter-2 OOS — stage-1 quality is a single sanity check (must beat trivial mean-RRP baseline on the fold's fit window) |

---

## Primary feature presets (iter-2 active set)

These are the five `is_primary=True` feature presets in
`src/autoresearch.py::FEATURE_PRESETS`. They're the configs that
produced a non-linear cell <15M val MSE in the iter-1 result matrix.

| Tag | Composition | Why |
|---|---|---|
| `cal_temp_apptemp_lag1-7_roll7` | calendar + temperature + apparent_temp + demand lag {1, 7} + 7-day rolling demand | Iter-1 champion's exact features (val MSE 8.88M) |
| `full` (leaky marker) | + observed RRP | Best `full` cell was 8.71M (C5 in the controlled bundle), but observed RRP is leaky — predicted-RRP variant replaces it |
| `cal_temp_apptemp_lag1-7` | drop the rolling mean | A3 in the controlled bundle (9.49M) — ablates roll7's marginal contribution |
| `cal_temp_lag1-7_roll7` | drop apparent_temp | Ablates apparent_temp's marginal contribution under the rolling-mean condition |
| `cal_temp_lag1-7` | drop both apparent_temp and rolling | A2 in the controlled bundle (10.07M) — the simplest preset that still beats the baseline materially |

Predicted-RRP variants of the first three are added explicitly:

| Tag | Stage-2 features (predicted RRP slotted in for observed RRP) |
|---|---|
| `cal_temp_apptemp_predRRP_lag1-7_roll7` | the iter-1 champion features + `rrp_predicted` from stage-1 |
| `cal_temp_predRRP_lag1-7_roll7` | without apparent_temp; isolates apparent_temp's effect under predicted RRP |
| `cal_temp_apptemp_predRRP_lag1-7` | without the rolling mean; isolates rolling's effect under predicted RRP |

---

## Outcomes from these narrowings

- **Compute per `run_autoresearch.py` roughly halves on candidate count.**
  Per-run candidates drop from ~13 (iter-1: 2 baselines + champion + 4
  challengers + many random untried) to ~7 (2 baselines + champion + 4
  challengers drawn from the narrowed primary set + 0 known_bad).

- **Each candidate's cost rises ~10×** because walk-forward fits the
  model 10 times per evaluation. **Net wall-clock per
  `run_autoresearch.py` invocation: ~5× iter-1**, but with much better
  signal per unit of compute (`mean ± std` vs point estimate). Expect
  1–3 minutes per invocation on the project's laptop; if a single
  invocation exceeds 10 minutes, lower `--n-challengers` or bump
  `min_train_size` to 1000 to drop fold count from 10 to ~7 (see
  `TWO_WEEK_PLAN.md`, "Compute budget guard").

- **The H2 question becomes *testable*.** In iter-1 we could not
  separate signal from leak; in iter-2 we can. Either direction of the
  predicted-RRP variant — supported, null, or rejected under the 5%
  minimum-effect-size criterion — is a publishable iter-2 result.
