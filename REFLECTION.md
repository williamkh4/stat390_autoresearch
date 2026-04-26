# Reflection — Iteration 1

A retrospective on the AutoResearch framework build. Written after the
first 5 dry-run experiments completed. Honest, not promotional.

---

## What the agent did well

**Locked design decisions early, in code.** Before writing any modeling
code, the validation metric (MSE), test split (final 365 days), and
baseline (seasonal naive) were committed via constants in
`src/metrics.py`, `src/split.py`, and `src/autoresearch.py`. This means
no number on any future leaderboard can quietly drift unless a constant
is intentionally edited. The `program.md` spec mirrors these decisions
in human-readable form. Drift is the most expensive bug in research code,
and this design makes it loud.

**Made the test set untouchable by construction.** `Splits.test` exists
as a dataclass field but is never passed into `run_loop`. To use it,
someone has to write new code, not just toggle a flag. This is a much
stronger guard against test-set leakage than relying on discipline.

**Treated the baseline question seriously.** When asked whether
`period=7` ignored yearly seasonality, the answer wasn't a hand-wavy
"trust me." It ran the comparison: `period=364` (yearly-aligned) cuts
MSE by 57% vs. `period=7`. That number is the strong baseline now, and
it's the right one to beat. Notably, `period=365` was *worse* than
`period=364` because 365 isn't a multiple of 7 — a real subtlety that
showed up empirically.

**Cross-run state landed cleanly.** `master_log.csv` and `champion.json`
plus `analyze_runs.py` give a single source of truth across runs without
a database. The champion is auto-promoted on improvement, which means
each new iteration is anchored against the right bar without any manual
bookkeeping.

**Surfaced silent failure modes proactively.** `validate_merged()` checks
for date gaps, NaNs in target, and merge-overlap shortfalls — exactly
the issues that distort MSE without crashing anything. The README's
"Failure modes" table separates silent (dangerous) from loud (visible)
issues, which is the categorization that actually helps under time
pressure.

**Caught and fixed real bugs during the build.** When the user pointed
out that `run_baseline.py` produced no artifact, the script was rewritten
to write JSON + CSV alongside the AutoResearch outputs. When the sklearn
imports were discovered to fire even for baseline-only flows, they were
moved inside `default_candidates()` and a separate `baseline_candidates()`
was extracted. Both fixes were small, but they would have cost real time
to discover later.

---

## What the agent did badly

**Initial `run_baseline.py` produced no artifact.** Documenting only via
stdout meant a re-run looked indistinguishable from a no-op. The user had
to ask before this got fixed. A reproducible analysis should always leave
a trail, and this should have been the default from line one.

**Hard-coded sklearn imports at module top.** The original
`autoresearch.py` did `from sklearn.linear_model import Ridge` at import
time, which broke `run_baseline.py` whenever sklearn wasn't installed —
even though the baselines themselves don't need sklearn. This was a
coupling bug that should have been caught the first time the dependency
graph was sketched.

**Default baseline initially understated the bar.** Iteration 1 shipped
with `seasonal_naive_7` as the only baseline. That number (MSE 141M) is
*easy* to beat — anything decent will. The stronger `seasonal_naive_364`
(MSE 60M) was added only after the user asked the right question. The
agent should have proposed both up front; the cost is two extra fits per
run, which is trivial.

**Validation window choice was arbitrary.** "180 days before test" was
asserted, not derived. A reasoned choice would have looked at the noise
floor (variance of the same model fit on different val windows) and
picked a window that was long enough for the signal to dominate but not
so long that train shrank meaningfully. Currently the only justification
is "half a seasonal cycle" — defensible but not load-bearing.

**Observed RRP as a feature is a semantic leak.** `gbm_full` and
`ols_full` use the *true* RRP at each forecast date as a feature. At
deployment you wouldn't have today's RRP — you'd have to predict it (the
charter's stage-1 model). The MSE reported for these candidates is
therefore optimistic. This is now flagged in the failure-modes table and
in §10 of `program.md`, but the candidate should have been called
something like `ols_full_with_observed_rrp` from the start to make the
caveat unmissable.

**Master log has no concurrency control.** `master_log.csv` is appended
with plain `df.to_csv(mode='a')`. Two `run_autoresearch.py` invocations
in parallel against the same `--results-dir` would interleave rows or
truncate each other. Unlikely in practice for a small research project,
but worth a comment in code; for now it's only a README footnote.

**Prose in the README grew incrementally.** Each user question added a
section, but the document wasn't refactored as it grew. A reader landing
on it cold has to scroll a lot to find the locked decisions. The new
`program.md` partially addresses this by giving the spec a separate home,
but the README itself could still be tighter.

---

## Findings from the 5 dry-run experiments

The dry runs used numpy-only OLS variants (so the framework could run
without sklearn) plus the two seasonal naives. Final state of the master
log under `experiments/results/dry_runs/`:

| candidate | best MSE | n_features | beats seasonal_naive_364? |
|---|---|---|---|
| `ols_full` | **39,872,301** | 25 | yes (−34%) |
| `seasonal_naive_364` | 60,410,779 | 0 | (baseline) |
| `ols_calendar` | 69,698,243 | 8 | no |
| `ols_calendar_temp_apptemp` | 92,072,320 | 21 | no |
| `ols_calendar_temp` | 98,265,550 | 18 | no |
| `seasonal_naive_7` | 141,138,184 | 0 | no |

Three things this surfaced that were **not** obvious in advance:

1. **Calendar features alone (OLS) lose to yearly-aligned seasonal naive.**
   `ols_calendar` (MSE 69.7M) is *worse* than `seasonal_naive_364`
   (60.4M). Eight calendar columns plus an intercept can't reproduce what
   recall of the same calendar position one year ago does for free. So
   the "real" signal in the dataset is more subtle than "season + day of
   week."

2. **Adding standard temperature features made OLS worse, not better.**
   `ols_calendar_temp` (98.3M) regressed sharply from `ols_calendar`
   (69.7M). The 10 temperature columns from the merged panel are highly
   correlated — they're essentially the same signal expressed three or
   four ways — and unregularized OLS struggles with that. This is exactly
   why the project charter prioritized tree-based models. A real Ridge
   with cross-validated alpha (instead of fixed alpha=1) would likely
   recover most of this.

3. **Apparent temperature partially helped, but didn't rescue the model.**
   `ols_calendar_temp_apptemp` (92.1M) improved over `ols_calendar_temp`
   (98.3M) by ~6%, supporting H1 *weakly*. But it still couldn't catch
   even calendar-only OLS, let alone the strong baseline. The
   apparent-temp finding needs to be re-tested with non-linear models
   before it can be reported with confidence.

The big win came from the full feature stack: adding lags `[1, 7]`,
`demand_roll7`, and observed RRP took the OLS from 92M to 39.9M — the
first model to beat the strong baseline, and the new champion.
*Caveat:* the RRP contribution there is inflated for the reason in the
"badly" section above. Iteration 2 should re-run `ols_full` without
RRP to isolate the lag contribution from the (leaky) RRP contribution.

---

## Failure modes encountered during this build

These are not hypothetical — they're the things that actually went wrong
or that the framework caught while it was being assembled. Cataloged
roughly in the order they appeared.

| # | Failure | How it surfaced | Resolution |
|---|---|---|---|
| 1 | `pip install scikit-learn` blocked by sandbox proxy | `Tunnel connection failed: 403 Forbidden` | Made the framework run without sklearn for baseline + numpy paths; added `numpy_models.py` |
| 2 | `run_baseline.py` produced no artifact | User noticed re-runs left no file | Rewrote to write `baseline_<timestamp>.json` + `.csv` |
| 3 | sklearn imports at module top broke baseline-only flow | `ModuleNotFoundError: No module named 'sklearn'` while running `run_baseline.py` | Lazy-imported sklearn inside `default_candidates()`; split out `baseline_candidates()` |
| 4 | Open-Meteo CSV had a 3-line metadata preamble | First `pd.read_csv` returned junk columns | `_find_timeseries_header` scans for `time,` and skips above it |
| 5 | Initial baseline was too easy to beat (period=7 only) | User asked the right ablation question; period=364 cut MSE by 57% | Added `seasonal_naive_364`; champion comparison now uses the *strongest* baseline |
| 6 | Inability to delete files in the sandbox | Could not reset `experiments/results/` for a clean rerun | Pointed dry runs at `experiments/results/dry_runs/` so traces stay self-contained |
| 7 | Adding standard temp features made OLS *worse* (E3) | MSE jumped from 69.7M to 98.3M | Diagnosed as multicollinearity in unregularized OLS; flagged for iteration 2 (cross-validated Ridge or trees) |
| 8 | Cross-run leaderboard had no good UI | "Where do I look?" question from user | Added `master_log.csv`, `champion.json`, and `analyze_runs.py` |
| 9 | Observed RRP used as a feature (semantic leak) | Spotted while writing the spec, not at runtime | Documented in failure-modes table; iteration 2 will swap in predicted RRP |
| 10 | Lag windows could in principle exceed `train` length | Considered while writing `features.py` | Long lags would cause `dropna()` to silently shrink the train set; mitigated by feature config defaults (`lags=[1,7]`) |
| 11 | `period=365` introduces day-of-week shift | Caught while comparing `period=7` to `period=365` for the user's question | Documented; baseline uses `period=364` instead |

The pattern: about half of these were caught by the framework's own
diagnostics or by trying to run code that I had written. The other half
came from the user asking the right question. That second category is
the more important one — without the user's "but what about period=365?"
the strong baseline would still be missing, and the loop's improvement
claims would be inflated.

---

## What would I do differently in iteration 2

1. **Replace observed RRP with predicted RRP.** Two-stage pipeline as
   the charter calls for. This is the single biggest correction the
   results need.
2. **Add cross-validated Ridge and a tree ensemble for the apparent-temp
   ablation.** OLS with `alpha=1` was the wrong tool for evaluating H1.
   The charter explicitly says "prioritize tree-based models for small
   data."
3. **Run each candidate ≥ 3 times and report mean ± std in the
   leaderboard.** Currently `analyze_runs.py --ablation` shows std but
   most candidates have only 1 sample. Repeated runs are cheap; the
   noise floor on MSE matters when the H1 effect size is in the
   single-digit-percent range.
4. **Tighten the README.** Move the locked-decisions table to the top,
   move detailed prose into `program.md` (mostly done), and shorten
   what's left.
5. **Add a tests/ directory.** At least a smoke test for
   `make_splits()` and `validate_merged()`. The framework is small
   enough that the tests would be small too, and they'd catch the
   coupling bug from item 3 of the failure-modes table.
