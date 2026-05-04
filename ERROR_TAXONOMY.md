# Error Taxonomy

Every issue encountered during this project, categorised under the four
failure-mode buckets used in coursework this week:

| category | what it means | where it shows up |
|---|---|---|
| **Signal Failure** | The loop runs, but no meaningful improvement appears across iterations. | The leaderboard plateaus; the auto-generator keeps producing rows that don't change the champion. |
| **Code Instability** | Crashes, inconsistent runs, or a broken pipeline that prevents reliable measurement. | Stack traces, missing artifacts, environment errors, candidates that "ran" but produced obviously wrong numbers. |
| **Evaluation Leakage** | The metric improves, but comparability is compromised — the evaluation setup shifted (or hid a bias). | Leaderboards that look better than they should; comparisons that don't survive a change of baseline / split / data definition. |
| **Agent Misbehavior** | The agent ignores rules or makes uncontrolled changes outside the intended scope. | Search scope silently shrinks/biases; conclusions drawn from partial data; decisions made that the user didn't sign off on. |

## Counts at a glance

| category | distinct issues |
|---|---:|
| Signal Failure | 2 |
| Code Instability | 5 |
| Evaluation Leakage | 6 |
| Agent Misbehavior | 3 |

Notably, **0 of 132 candidate fits raised a runtime exception** in the
auto-runs. Most "failures" below are *quiet*: the loop completed, but
something further upstream or downstream of the fit was wrong.

---

## Signal Failure

**S1. Champion plateau.** The current MLP champion (val MSE 8.88M) was set
in run 7. Runs 8–19 (84 candidate fits) produced no further improvement.
That plateau is information — but the loop has no exit criterion, so it
keeps drawing new candidates forever. Without a noise floor (see L4) we
also can't tell whether the plateau is "champion is genuinely near
optimal" or "everything we draw is within noise of the leader."

**S2. Linear-model corner of the search space yields nothing.** Across
every feature preset that has been tried with `numpy_ols` and `ridge`,
the best validation MSE bottoms out around 40M — about 4× worse than
the best tree/MLP. No combination of feature toggles or alpha values
moved the needle. The loop kept exploring linear candidates because
they were untried in the discrete space, but none of those rows ever
contributed an improvement.

---

## Code Instability

**C1. Open-Meteo CSV had a 3-line metadata preamble.** `pd.read_csv`
returned junk columns until `data_loader._find_timeseries_header` was
added to scan for `time,` and skip above it. *Resolved.*

**C2. sklearn imports at module top broke the baseline-only flow.**
Original `autoresearch.py` imported sklearn at import time, so
`run_baseline.py` errored on machines without sklearn. Lazy-imported
sklearn inside `_make_factory()`; split `baseline_candidates()` out as
a sklearn-free function. *Resolved.*

**C3. `run_baseline.py` originally produced no artifact.** Re-runs left
no on-disk evidence — re-running the script looked indistinguishable
from a no-op. Now writes `baseline_<utc_timestamp>.json` + `.csv`.
*Resolved.*

**C4. `master_log.csv` has no concurrency control.** Two
`run_autoresearch.py` invocations against the same `--results-dir` would
interleave or truncate rows. Low risk for a small research project
running on one machine, but the framework would mis-measure if anything
ever ran in parallel. *Open, low priority.*

**C5. MLP catastrophic non-convergence (silent training failure).**
Three cells in the result matrix show MLPs with absurd MSE:
`cal_temp_lag1-7 × mlp` = **1378M**, `cal_temp × mlp` = **627M**,
`cal_temp_apptemp_lag1-7-14_roll7-28 × mlp` = **618M**. None of these
crashed; the fit returned a model and the predictions were ~10× worse
than a constant predictor would be. The pipeline produced a number,
the loop accepted it, and the leaderboard listed it next to good
candidates. *Open;* `EXPERIMENTS.md` Series C (lr sweep) is the start
of the diagnostic.

---

## Evaluation Leakage

**L1. Observed RRP used as a feature (semantic leak).** Any candidate
on the `full` feature preset reads the *observed* RRP at each forecast
date. At deployment you wouldn't have today's RRP — you'd have to
predict it. The MSE for those candidates is therefore optimistically
biased. The current champion happens to drop RRP, so the leaderboard's
top row is unaffected, but the H2 (price-signal) hypothesis cannot be
tested honestly until the two-stage pipeline lands. *Documented,
deferred.*

**L2. Initial baseline (period=7) was too easy to beat → comparability
shifted mid-project.** Iteration 0 shipped only `seasonal_naive_7`
(MSE 141M), and any model trivially beat it. Adding the
yearly-aligned `seasonal_naive_364` (MSE 60M) turned the bar from
trivial to meaningful, but every prior-iteration claim about "beating
the baseline" was retroactively a different statement. *Resolved going
forward;* prior numbers in REFLECTION.md and program.md should be
read with this in mind.

**L3. `period=365` introduces a day-of-week shift.** A naive
yearly-recall baseline at period 365 would lookup last year's same
date but a different day-of-week. Empirical comparison vs `period=364`
showed the day-of-week shift inflates MSE measurably. Baseline locked
at `period=364` (52×7) instead. *Resolved.*

**L4. Variance estimates are uninformative.** `analyze_runs.py
--ablation` reports `std = 0.0` for every multi-run candidate because
the framework is deterministic given fixed splits and fixed seeds.
The 21% improvement that promoted the MLP champion is reportable but
not statistically defensible — we cannot tell how much the MSE moves
under a slightly different val cut or train resample. *Open; co-dominant
failure mode together with L6* (see `FAILURE_ANALYSIS_MEMO.docx`).

**L5. Test-set evaluation runner now exists, gated on noise floor.**
`run_test_evaluation.py` re-fits the champion on `train + val` and
predicts on the locked 365-day test window once. The framework can now
report a test number with all three metrics (`mse_demand`,
`rmse_demand`, `mae_demand`). The runner refuses repeat evaluations
of the same champion by default to keep the held-out set held-out.
`run_autoresearch.py` additionally gained `--evaluate-on-test` (every
candidate scored on val and test simultaneously, recommended for
visibility) and `--promote-on=test` (opt-in, with a multi-line
warning, because it burns the test set as a tuning surface). *Status:
runner shipped; recommended to land L4 (noise floor) before
publishing the test number.*

**L6. Severe val→test distribution shift in the locked test window.**
The test window (2019-10-08 → 2020-10-06) covers the start of the
COVID-19 demand shock in Australia. Both seasonal-naive baselines
degrade massively from val to test:

| baseline | val MSE | test MSE | val→test factor |
|---|---:|---:|---:|
| `seasonal_naive_364` | 60.41M | 188.09M | 3.11× |
| `seasonal_naive_7`   | 141.14M | 206.58M | 1.46× |

These are deterministic numbers (no model fitting noise), so the gap
*is* real — the test data is harder than the val data, full stop. Any
sophisticated model that beats the baselines on val will also see its
absolute MSE rise substantially on test for the same reason. The
*relative* ranking should mostly hold (a sandbox run of a numpy_ols
stand-in for the MLP champion still beat `seasonal_naive_364` on test
by ~73%), but cross-iteration comparisons of absolute test MSE will be
misleading unless the shift is acknowledged. *Open;* most direct
mitigation is a sensitivity check using a non-COVID test window
(e.g. last 365 days *before* 2019-10-08), even if only as a secondary
report alongside the locked one.

---

## Agent Misbehavior

**A1. Auto-generator concentrates on the champion's neighborhood.**
The candidate-generation rule prefers one-knob mutations of the current
champion's *feature config*, paired with random model specs. After GBM
became champion in run 1, runs 2–6 over-explored GBM territory; after
the MLP took over in run 7, the same dynamic kicked in around MLP. The
agent isn't violating any rule, but it's making an uncontrolled
exploration choice the user didn't explicitly request. Mitigation:
reserve ≥1 challenger per run for "different model class than current
champion." *Open.*

**A2. `--promote-on=test` is a methodology trapdoor.** The new flag
allows the auto-loop to use test MSE for champion selection. Used
casually it would convert the locked test set into a second validation
set within a single run, and across runs it would burn the held-out
window entirely. The runner prints a multi-line warning when the flag
is set, but a one-line warning is not the same as a guard rail; a
distracted user could end up making selection decisions on the test
set without re-reading the warning. Mitigation: keep the flag opt-in,
and the post-run report should clearly mark champion records with
`metric_source: "test"` so any later report can flag promotions made
under that policy. *Status: design choice, accepted with disclaimer.*

**A3. Drew conclusions from partial data (historical).** Earlier in
this project the agent wrote a REFLECTION.md based on a 6-run snapshot
of `master_log.csv`, not the canonical 19-run record. The conclusion
("MLP underperformed") was a false negative caused by snapshot scope,
and it would have stuck in the project record had the user not
uploaded the full data later. Mitigation: always print "this analysis
is based on N runs / M candidates" at the top of any generated
reflection, and refuse to write conclusions without a sanity-check on
history completeness.

---

## Cross-cutting observation

The dominant pattern across the four buckets is that **most failures
are quiet, not loud.** Code Instability has 5 entries but only 3
actually crashed; the other 2 (C4 concurrency, C5 silent MLP) produced
incorrect output without raising. Every Evaluation Leakage and Agent
Misbehavior item is quiet by definition.

The two most consequential remaining items — L4 (no noise floor) and
L6 (val→test distribution shift) — are now *co-dominant*. L4 means we
can't say whether 8.88M is meaningfully better than 9.10M. L6 means the
absolute test number we eventually report will be inflated by the
COVID-era regime change, regardless of model quality. Both need to
land before the project closes; see `FAILURE_ANALYSIS_MEMO.docx` for
the joint plan.
