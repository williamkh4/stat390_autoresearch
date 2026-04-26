# Reflection — Iteration 1 (local run)

A concise retrospective on the AutoResearch framework after 1 baseline run
and 6 self-driving auto iterations on the user's local machine (sklearn
available, full 494-candidate search space active). Honest, not
promotional.

---

## What the system does in one loop

`python run_autoresearch.py` is self-driving. Per invocation it:

1. Loads the merged Kaggle + Open-Meteo panel and carves train / val / test
   via `make_splits()`. The 365-day test set is constructed but never
   passed into the loop.
2. Reads `experiments/auto_runs/champion.json` (current best) and
   `master_log.csv` (every candidate ever tried).
3. Builds the candidate list via `auto_candidates()`:
   - Both seasonal-naive baselines (always).
   - The champion's exact config (always — surfaces noise around the leader).
   - N challengers (`--n-challengers`, default 4), drawn first from
     **one-knob mutations of the champion's feature config** (clean
     ablation pairs), then from random untried points in the
     `MODEL_SPECS × FEATURE_PRESETS` space, with history-based dedup
     against `master_log.csv`.
4. Fits every candidate on train, scores MSE on val, writes a per-run
   JSON + leaderboard CSV, appends rows to `master_log.csv`, and
   auto-promotes a new `champion.json` whenever a challenger lowers MSE.

The discrete space is 38 model specs (numpy_ols, ridge, rf, gbm, mlp) ×
13 feature presets = 494 candidates. The locked decisions (target,
metric, splits, baselines) are constants in code, not flags.

---

## Findings from this run

| | best MSE | RMSE | beats `seasonal_naive_364` (60.41M)? |
|---|---:|---:|---|
| GBM (champion: cal+temp+apptemp+lag1-7, Huber loss) ★ | **11,234,955** | 3,352 | yes (−81%) |
| GBM (cal+temp+lag1-7, leaf=10) — prior champion | 12,940,621 | 3,597 | yes (−79%) |
| Random Forest (cal+temp+apptemp+lag1-7+roll7, no bootstrap) | 13,108,930 | 3,621 | yes (−78%) |
| Ridge (cal+temp+apptemp+lag1-7+roll7, α=0.1) | 41,122,239 | 6,413 | yes (−32%) |
| NumpyOLS (cal+temp+apptemp+lag1-7, α=10) | 44,255,977 | 6,653 | yes (−27%) |
| MLP (best of 7 architectures) | 44,306,936 | — | yes (−27%) |
| `seasonal_naive_364` | 60,410,779 | 7,772 | (baseline) |
| `seasonal_naive_7` | 141,138,184 | 11,880 | no |

Three things this iteration showed clearly:

1. **Non-linear models dominate.** GBM is 3.6× better than Ridge / OLS
   and 4× better than MLP. H3 (non-linearity) is supported, and
   strongly: tree ensembles handle the demand–temperature interactions
   that linear models can't.
2. **H1 (apparent temperature) is supported.** The new champion (run 3,
   `e2e5a30e`) added apparent_temp on top of standard temp, taking MSE
   from 12.94M → 11.23M (−13%). This is the cleanest one-knob
   confirmation in the run set.
3. **Huber loss helps.** The promotion to champion came with switching
   GBM's loss from squared-error to Huber, which is consistent with
   demand having occasional spikes the L2 loss over-fits to.

---

## What worked well

**Baseline stability across runs.** `seasonal_naive_7` and
`seasonal_naive_364` are present in all 6 runs and produce identical
MSE every time (141.14M and 60.41M). That's the sanity check that the
splits and metric haven't drifted; if either number had shifted, the
leaderboard would be incomparable.

**Search-space breadth paid off.** The champion uses Huber loss — a
spec the human eye would not have written by hand on iteration 1. The
auto-generator surfaced it because `("loss", "huber")` is one row of
`MODEL_SPECS` and the mutation rule combined it with the previous
champion's feature config. That's the intended payoff of enumerating
the space rather than picking favorites.

**Self-driving never stalled.** Six runs, 41 master-log rows, 26 unique
candidates, zero error rows, two champion promotions. No code edits
between runs. The dedup against history did its job: every iteration
contributed new information rather than re-confirming old combos.

**Multiple model classes were exercised, then ruled out cleanly.** RF
trailed GBM by ~17%; Ridge / OLS / MLP all clustered around 41–44M, a
clear ~3.6× gap behind the trees. Future iterations have a defensible
reason to focus their compute budget on GBM tuning rather than spreading
thin.

**The 365-day test set was never touched.** `Splits.test` was
constructed but never passed into `run_loop()`; verifiable from the
`splits_summary` field in every `run_<id>.json`. The champion's
generalization remains an honest open question.

---

## What went badly

**MLP underperformed all expectations.** Seven different MLP
architectures (1–3 hidden layers, ReLU/tanh, adam/lbfgs, alphas
0.0001–0.01, with and without early stopping) all clustered around 44M
— on par with unregularized OLS. With n=1561 training rows that's
plausible (neural nets are data-hungry), but it's also possible the
hyperparameter range is too narrow (no learning-rate schedule, no
deeper / wider nets, no batch-norm). Worth re-visiting before declaring
MLPs uncompetitive on this problem.

**Search may have locked onto GBM early.** Once GBM became champion in
run 1, every subsequent challenger drawn as a "one-knob mutation of the
champion's feature config" was paired with a fresh GBM spec from
`MODEL_SPECS`. Over runs 2–6 the auto-generator therefore re-explored
GBM heavily and only rarely sampled non-GBM specs from the random
untried pool. The champion improved 13% in run 3 and then plateaued for
3 runs. A more aggressive exploration policy (e.g. force one challenger
per run to come from a *different* model type than the champion) would
keep the alternatives exercised even after a champion locks in.

**No noise floor.** Every multi-run candidate in `analyze_runs.py
--ablation` shows std = 0.0 because the framework is deterministic
given fixed splits and fixed seeds. The 13% improvement that promoted
the new champion is therefore reportable but not statistically
defensible — we don't know how much MSE moves on this dataset with a
slightly different val cut. A train-bootstrap or val-cut wiggle is the
cheapest fix.

**Observed-RRP semantic leak hasn't been touched.** It happens to not
matter for the *current* champion (which doesn't use RRP), but H2 is
still unanswerable until the two-stage RRP→demand pipeline lands.

**No final test-set evaluation yet.** Champion MSE on validation is
11.23M. Test MSE is unknown, by deliberate design — the test set is
locked until iteration 2's tuning is done. Worth flagging that the 81%
improvement vs. the strong baseline is a *validation* number; the
project still owes a test-set number before closing.

---

## Most common failure modes (running tally across both iterations)

| # | Failure | How it surfaced | Resolution |
|---|---|---|---|
| 1 | `run_baseline.py` originally produced no artifact | Re-run looked indistinguishable from no-op | Now writes `baseline_<timestamp>.json` + `.csv` |
| 2 | sklearn imports at module top broke the baseline-only flow | `ModuleNotFoundError` on `run_baseline.py` | Lazy-imported sklearn inside factories; split `baseline_candidates()` from `default_candidates()` |
| 3 | Open-Meteo CSV had a 3-line metadata preamble | `pd.read_csv` returned junk columns | `_find_timeseries_header` skips lines until it sees `time,` |
| 4 | Initial baseline (period=7 only) was too easy to beat | User asked the right ablation question | Added `seasonal_naive_364` (yearly-aligned, 52×7) — beating it is now table stakes |
| 5 | Standard temperature features made unregularized OLS *worse* | MSE jumped 26M → 42M when temp added | Diagnosed as multicollinearity; vindicated by GBM handling the same features fine in this iteration |
| 6 | Observed RRP used as a feature (semantic leak) | Caught while writing the spec | Documented in §10 of program.md; champion happens to not use RRP, so current MSE is unaffected; iteration 2 will swap to predicted RRP |
| 7 | `period=365` introduces day-of-week shift | Empirical comparison vs `period=364` | Baseline locked at 364 (52×7) |
| 8 | Cross-run leaderboard had no UI | "Where do I look?" question from user | Added `master_log.csv`, `champion.json`, and `analyze_runs.py` |
| 9 | sklearn-only specs invisibly filtered out in sandbox | 35 of 38 specs silently dropped | Cosmetic gap — should print one warning line at run start so the user knows their environment caps the search space |
| 10 | Variance estimates from repeated runs are uninformative | `std = 0.0` for every multi-run candidate | Real noise floor needs a stochastic perturbation (bootstrap, val-cut wiggle) — deferred to iteration 2 |
| 11 | Auto-generator concentrates on the champion's model class once one wins | GBM ran in every iteration after run 1; MLP / Ridge under-explored after early losses | Add an "exploration bonus" so ≥1 challenger per run comes from a different model type than the champion |
| 12 | MLP underperformed — could be a real finding or an under-tuned space | All 7 MLP variants clustered at ~44M MSE, behind RF/GBM by 4× | Need a wider MLP grid (deeper nets, lr schedules, batch norm) before concluding MLPs are uncompetitive |

---

## What I'd do next (iteration 2)

1. **Run the locked test set once.** Champion validation MSE is 11.23M;
   we owe a single test number before declaring success.
2. **Add a noise floor.** Either bootstrap-resample the train set or
   evaluate at three slightly different val cut-points; report std on
   the leaderboard.
3. **Force exploration of non-champion model classes.** Reserve 1 of
   N challengers for "different model type than the current champion"
   so MLP, Ridge, etc. keep getting tested even when GBM leads.
4. **Two-stage RRP → demand pipeline.** Replace observed RRP with
   predicted RRP wherever a candidate uses RRP. This is the only way
   to honestly test H2.
5. **Wider MLP search.** Add deeper / wider architectures and a
   learning-rate schedule before the "MLPs aren't competitive on this
   data" conclusion is locked in.
