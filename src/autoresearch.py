"""
AutoResearch loop.

Iter-1 contract:
  One "run" = fit and score every candidate on a single 180-day validation
  hold-out, write a leaderboard, return the best candidate's config.

Iter-2 contract (the version in this file):
  One "run" = fit and score every candidate across ~10 walk-forward folds
  (expanding train, 180-day val window, 90-day step, train >= 730 days),
  report `mean ± std` for every metric, write the leaderboard, return the
  champion. The locked 365-day test set is still never touched here.

Iter-2 also adds:

  * Noise-aware champion promotion:
      promote iff  mean_challenger + std_challenger
                       < mean_champion   − std_champion
    (the deterministic "lower MSE wins" rule of iter-1 over-promoted on noise.)

  * Two-stage RRP pipeline:
      FeatureConfig.use_predicted_rrp = True triggers
      src.predict_rrp.materialize_predicted_rrp() per-fold (stage-1 fit
      only on the fold's train rows) and feeds the predicted RRP into
      the stage-2 demand model as a regular feature. Observed-RRP
      (FeatureConfig.use_rrp) is retained but marked leaky.

  * Search-space narrowing:
      Each ModelSpec carries `is_sanity_only` (numpy_ols / ridge), `known_bad`
      (the 3 catastrophic MLP configs in iter-1's result matrix). Each
      FeaturePreset carries `is_primary` (5 winners) and `is_leaky_marker`
      (observed-RRP presets, deprioritised). The auto-generator prefers
      primary + non-sanity + non-known-bad.

Budget accounting per run:
  * runtime_sec per candidate and total wall-clock for the run
  * fit_count = number of `(candidate, fold)` pairs scored
  * peak memory and gpu are still out of scope
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Any, Tuple, Union
import importlib
import json
import math
import platform
import random
import time
import uuid
import warnings

import numpy as np
import pandas as pd

from .baselines import SeasonalNaive
from .features import FeatureConfig, build_features, PREDICTED_RRP_COL
from .metrics import PRIMARY_METRIC_NAME, score_all
from .numpy_models import NumpyOLS
from .split import Splits, WalkForwardFold


# ============================================================================
# Candidate definitions
# ============================================================================

@dataclass
class Candidate:
    name: str
    feature_config: FeatureConfig
    model_factory: Callable[[], Any]   # returns a fresh sklearn-like estimator
    is_baseline: bool = False
    # Search-space provenance / triage flags. Set by the auto-generator from
    # the ModelSpec / FeaturePreset metadata; controlled-experiment scripts
    # leave them at their defaults.
    is_sanity_only: bool = False
    known_bad: bool = False

    def describe(self) -> str:
        return f"{self.name} | features[{self.feature_config.describe()}]"


def baseline_candidates() -> List[Candidate]:
    """Baseline reference models. Depend only on numpy/pandas, not sklearn."""
    empty_fc = FeatureConfig(
        use_calendar=False, use_temp=False, use_apparent_temp=False,
        use_rrp=False, use_predicted_rrp=False,
        demand_lags=[], rolling_windows=[],
    )
    return [
        Candidate(
            name="seasonal_naive_7",
            feature_config=empty_fc,
            model_factory=lambda: SeasonalNaive(period=7),
            is_baseline=True,
        ),
        Candidate(
            name="seasonal_naive_364",
            feature_config=empty_fc,
            model_factory=lambda: SeasonalNaive(period=364),
            is_baseline=True,
        ),
    ]


def default_candidates() -> List[Candidate]:
    """Seed set for the first AutoResearch iteration = baselines + real models.

    Kept intact for backwards compatibility with the iter-1 README. New
    work goes through `auto_candidates()` which reads the narrowed
    iter-2 search space.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    return baseline_candidates() + [
        Candidate(
            name="ridge_cal_temp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=False,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: Ridge(alpha=1.0, random_state=0),
        ),
        Candidate(
            name="ridge_cal_temp_apptemp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=True,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: Ridge(alpha=1.0, random_state=0),
        ),
        Candidate(
            name="rf_cal_temp_apptemp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=True,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: RandomForestRegressor(
                n_estimators=200, max_depth=None, random_state=0, n_jobs=-1,
            ),
        ),
        Candidate(
            name="gbm_full",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=True,
                use_rrp=True, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0,
            ),
        ),
    ]


# ============================================================================
# Search space (iter-2 narrowed)
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    """Model class + kwargs + provenance flags for the iter-2 search space."""
    model_type: str
    kwargs: Tuple[Tuple[str, Any], ...]
    is_sanity_only: bool = False    # kept for smoke tests; not part of champion comparison
    known_bad: bool = False         # produced >500M val MSE in iter-1; auto-gen skips

    @property
    def key(self) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
        return (self.model_type, self.kwargs)


@dataclass(frozen=True)
class FeaturePreset:
    """Feature config + tag + provenance flags for the iter-2 search space."""
    tag: str
    config: FeatureConfig
    is_primary: bool = False        # one of the 5 top-performing presets in iter-1
    is_leaky_marker: bool = False   # uses observed RRP (deprecated; predicted-RRP variant exists)
    is_two_stage: bool = False      # uses FeatureConfig.use_predicted_rrp


# --- Model specs ----------------------------------------------------------
#
# Linear models retained as one numpy_ols + one ridge sanity row each.
# Tree ensembles + MLP make up the active champion-comparison space. Three
# MLP configs that produced catastrophic val MSE in iter-1 are flagged
# known_bad and skipped by the auto-generator.
MODEL_SPECS: List[ModelSpec] = [
    # --- Linear (sanity-only in iter-2) -------------------------------------
    ModelSpec("numpy_ols", (("alpha", 1.0),),                         is_sanity_only=True),
    ModelSpec("ridge",     (("alpha", 1.0), ("random_state", 0)),     is_sanity_only=True),

    # --- Random Forest ------------------------------------------------------
    ModelSpec("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 1),
                     ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ModelSpec("rf", (("n_estimators", 200), ("max_depth", 10),   ("min_samples_leaf", 1),
                     ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ModelSpec("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 5),
                     ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ModelSpec("rf", (("n_estimators", 500), ("max_depth", None), ("min_samples_leaf", 5),
                     ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),

    # --- Gradient Boosting --------------------------------------------------
    ModelSpec("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
                      ("subsample", 1.0), ("random_state", 0))),
    ModelSpec("gbm", (("n_estimators", 300), ("max_depth", 5), ("learning_rate", 0.05),
                      ("subsample", 1.0), ("random_state", 0))),
    ModelSpec("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
                      ("subsample", 1.0), ("loss", "huber"), ("random_state", 0))),
    ModelSpec("gbm", (("n_estimators", 500), ("max_depth", 3), ("learning_rate", 0.05),
                      ("subsample", 0.8), ("n_iter_no_change", 10),
                      ("validation_fraction", 0.15), ("random_state", 0))),

    # --- Multi-layer perceptron --------------------------------------------
    # The iter-1 champion's exact spec leads.
    ModelSpec("mlp", (("hidden_layer_sizes", (128, 64)),  ("activation", "relu"), ("solver", "adam"),
                      ("alpha", 0.001), ("learning_rate_init", 0.01),
                      ("max_iter", 500), ("random_state", 0))),
    ModelSpec("mlp", (("hidden_layer_sizes", (64, 32)),   ("activation", "relu"), ("solver", "adam"),
                      ("alpha", 0.001), ("learning_rate_init", 0.001),
                      ("max_iter", 500), ("random_state", 0))),
    ModelSpec("mlp", (("hidden_layer_sizes", (64, 32)),   ("activation", "relu"), ("solver", "adam"),
                      ("alpha", 0.001),  ("learning_rate_init", 0.001),
                      ("max_iter", 1000), ("early_stopping", True),
                      ("validation_fraction", 0.15), ("random_state", 0))),
    # known_bad: these three produced >500M val MSE in iter-1 result matrix.
    ModelSpec("mlp", (("hidden_layer_sizes", (64, 32)),   ("activation", "relu"), ("solver", "adam"),
                      ("alpha", 0.01),   ("learning_rate_init", 0.001),
                      ("max_iter", 500), ("random_state", 0)), known_bad=True),
    ModelSpec("mlp", (("hidden_layer_sizes", (64, 32)),   ("activation", "tanh"), ("solver", "adam"),
                      ("alpha", 0.0001), ("learning_rate_init", 0.001),
                      ("max_iter", 500), ("random_state", 0)), known_bad=True),
    ModelSpec("mlp", (("hidden_layer_sizes", (128, 64, 32)),("activation", "relu"), ("solver", "adam"),
                      ("alpha", 0.0001), ("learning_rate_init", 0.001),
                      ("max_iter", 500), ("random_state", 0)), known_bad=True),
]


# --- Feature presets ------------------------------------------------------
def _fc(use_calendar=False, use_temp=False, use_apparent_temp=False,
        use_rrp=False, use_predicted_rrp=False,
        demand_lags=None, rolling_windows=None) -> FeatureConfig:
    return FeatureConfig(
        use_calendar=use_calendar, use_temp=use_temp,
        use_apparent_temp=use_apparent_temp, use_rrp=use_rrp,
        use_predicted_rrp=use_predicted_rrp,
        demand_lags=list(demand_lags or []),
        rolling_windows=list(rolling_windows or []),
    )


FEATURE_PRESETS: List[FeaturePreset] = [
    # --- Primary (iter-2 champion-comparison set) ---------------------------
    # 5 presets that achieved a non-linear cell <15M val MSE in iter-1's
    # analysis/result_matrix.md. These are the active search surface.
    FeaturePreset("cal_temp_apptemp_lag1-7_roll7",
                  _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                      demand_lags=[1, 7], rolling_windows=[7]),
                  is_primary=True),
    FeaturePreset("cal_temp_apptemp_lag1-7",
                  _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                      demand_lags=[1, 7]),
                  is_primary=True),
    FeaturePreset("cal_temp_lag1-7_roll7",
                  _fc(use_calendar=True, use_temp=True,
                      demand_lags=[1, 7], rolling_windows=[7]),
                  is_primary=True),
    FeaturePreset("cal_temp_lag1-7",
                  _fc(use_calendar=True, use_temp=True, demand_lags=[1, 7]),
                  is_primary=True),
    # Observed-RRP "full" preset is leaky (L1); kept for replay only, the
    # predicted-RRP variant below is the iter-2 honest replacement.
    FeaturePreset("full",
                  _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                      use_rrp=True, demand_lags=[1, 7], rolling_windows=[7]),
                  is_primary=True, is_leaky_marker=True),

    # --- Two-stage predicted-RRP variants (iter-2 H2 surface) ---------------
    FeaturePreset("cal_temp_apptemp_predRRP_lag1-7_roll7",
                  _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                      use_predicted_rrp=True,
                      demand_lags=[1, 7], rolling_windows=[7]),
                  is_primary=True, is_two_stage=True),
    FeaturePreset("cal_temp_predRRP_lag1-7_roll7",
                  _fc(use_calendar=True, use_temp=True,
                      use_predicted_rrp=True,
                      demand_lags=[1, 7], rolling_windows=[7]),
                  is_primary=True, is_two_stage=True),
    FeaturePreset("cal_temp_apptemp_predRRP_lag1-7",
                  _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                      use_predicted_rrp=True, demand_lags=[1, 7]),
                  is_primary=True, is_two_stage=True),

    # --- Non-primary (kept in registry, deprioritised by auto-gen) ----------
    FeaturePreset("cal", _fc(use_calendar=True)),
    FeaturePreset("cal_lag1-7", _fc(use_calendar=True, demand_lags=[1, 7])),
    FeaturePreset("cal_lag1-7_roll7",
                  _fc(use_calendar=True, demand_lags=[1, 7], rolling_windows=[7])),
    FeaturePreset("cal_temp", _fc(use_calendar=True, use_temp=True)),
    FeaturePreset("cal_temp_apptemp",
                  _fc(use_calendar=True, use_temp=True, use_apparent_temp=True)),
    FeaturePreset("cal_temp_apptemp_lag1-7-14_roll7-28",
                  _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                      demand_lags=[1, 7, 14], rolling_windows=[7, 28])),
]


def _sklearn_available() -> bool:
    try:
        importlib.import_module("sklearn")
        return True
    except ImportError:
        return False


def _model_spec_compatible(model_type: str) -> bool:
    if model_type == "numpy_ols":
        return True
    if model_type in {"ridge", "rf", "gbm", "mlp"}:
        return _sklearn_available()
    return False


def _format_kwargs(kwargs: Tuple[Tuple[str, Any], ...]) -> str:
    """Stable suffix for the candidate name, e.g. 'alpha1.0_n200'."""
    parts: List[str] = []
    short = {
        "alpha": "alpha",
        "n_estimators": "n",
        "max_depth": "depth",
        "learning_rate": "lr",
        "subsample": "sub",
        "min_samples_leaf": "leaf",
        "min_samples_split": "split",
        "max_features": "feat",
        "max_leaf_nodes": "lnodes",
        "loss": "loss",
        "n_iter_no_change": "patience",
        "validation_fraction": "valf",
        "bootstrap": "boot",
        "hidden_layer_sizes": "hls",
        "activation": "act",
        "solver": "solver",
        "learning_rate_init": "lrinit",
        "max_iter": "iter",
        "early_stopping": "earlystop",
    }
    skip = {"random_state", "n_jobs"}
    for k, v in kwargs:
        if k in skip:
            continue
        key = short.get(k, k)
        if isinstance(v, tuple):
            v_str = "x".join(str(x) for x in v)
        else:
            v_str = str(v)
        parts.append(f"{key}{v_str}")
    return "_".join(parts)


def _config_name(model_type: str, model_kwargs: Tuple[Tuple[str, Any], ...],
                 preset_tag: str) -> str:
    suffix = _format_kwargs(model_kwargs)
    return f"{model_type}__{preset_tag}__{suffix}" if suffix else f"{model_type}__{preset_tag}"


def _make_factory(model_type: str,
                  model_kwargs: Tuple[Tuple[str, Any], ...]) -> Callable[[], Any]:
    """Build a zero-arg factory for a given (model_type, kwargs) spec."""
    kwargs = dict(model_kwargs)

    if model_type == "numpy_ols":
        def factory() -> Any:
            return NumpyOLS(alpha=float(kwargs.get("alpha", 0.0)))
        return factory

    if model_type == "ridge":
        def factory() -> Any:
            from sklearn.linear_model import Ridge
            return Ridge(**kwargs)
        return factory

    if model_type == "rf":
        def factory() -> Any:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**kwargs)
        return factory

    if model_type == "gbm":
        def factory() -> Any:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**kwargs)
        return factory

    if model_type == "mlp":
        def factory() -> Any:
            from sklearn.neural_network import MLPRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            return Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(**kwargs)),
            ])
        return factory

    raise ValueError(f"Unknown model_type: {model_type!r}")


def _full_search_space() -> List[Candidate]:
    """Every (ModelSpec x FeaturePreset) candidate compatible with the host."""
    out: List[Candidate] = []
    for spec in MODEL_SPECS:
        if not _model_spec_compatible(spec.model_type):
            continue
        for preset in FEATURE_PRESETS:
            name = _config_name(spec.model_type, spec.kwargs, preset.tag)
            out.append(Candidate(
                name=name,
                feature_config=preset.config,
                model_factory=_make_factory(spec.model_type, spec.kwargs),
                is_baseline=False,
                is_sanity_only=spec.is_sanity_only,
                known_bad=spec.known_bad,
            ))
    return out


def _read_history_names(results_dir: Path) -> set[str]:
    log_path = Path(results_dir) / MASTER_LOG_NAME
    if not log_path.exists():
        return set()
    try:
        df = pd.read_csv(log_path, usecols=["candidate_name"])
    except Exception:
        return set()
    return set(df["candidate_name"].dropna().astype(str).tolist())


def _fc_signature(fc: FeatureConfig) -> Tuple:
    return (
        fc.use_calendar, fc.use_temp, fc.use_apparent_temp, fc.use_rrp,
        getattr(fc, "use_predicted_rrp", False),
        tuple(fc.demand_lags), tuple(fc.rolling_windows),
    )


def _knob_distance(fc_a: FeatureConfig, fc_b: FeatureConfig) -> int:
    sig_a = _fc_signature(fc_a)
    sig_b = _fc_signature(fc_b)
    return sum(1 for x, y in zip(sig_a, sig_b) if x != y)


def _champion_feature_config(champion: Dict[str, Any] | None) -> FeatureConfig | None:
    if not champion:
        return None
    fc = champion.get("feature_config")
    if not fc:
        return None
    return FeatureConfig(
        use_calendar=bool(fc.get("use_calendar", False)),
        use_temp=bool(fc.get("use_temp", False)),
        use_apparent_temp=bool(fc.get("use_apparent_temp", False)),
        use_rrp=bool(fc.get("use_rrp", False)),
        use_predicted_rrp=bool(fc.get("use_predicted_rrp", False)),
        demand_lags=list(fc.get("demand_lags", []) or []),
        rolling_windows=list(fc.get("rolling_windows", []) or []),
    )


def auto_candidates(
    prev_champion: Dict[str, Any] | None,
    results_dir: Path | str,
    n_challengers: int = 4,
    seed: int | None = None,
) -> List[Candidate]:
    """Build a self-driving candidate list for one iter-2 iteration.

    Iter-2 strategy (priority order):

      1. **Baselines** -- always included.
      2. **Champion config re-run** -- always included (variance check on the
         current best).
      3. **One-knob mutations of the champion's feature_config** -- the
         cleanest ablations, drawn from the *primary* preset set first,
         skipping known_bad model specs and sanity-only entries.
      4. **Untried primary candidates** -- random draw from
         (primary preset) x (non-sanity-only, non-known-bad ModelSpec).
      5. **Untried non-primary fallback** -- only if step 4 is exhausted.

    Names already in `master_log.csv` are skipped from steps 3, 4, 5.
    """
    candidates: List[Candidate] = list(baseline_candidates())
    history = _read_history_names(Path(results_dir))
    full_space = _full_search_space()

    # Index helpers
    primary_tags = {p.tag for p in FEATURE_PRESETS if p.is_primary}
    leaky_tags = {p.tag for p in FEATURE_PRESETS if p.is_leaky_marker}

    def is_primary_candidate(c: Candidate) -> bool:
        preset_tag = c.name.split("__")[1] if "__" in c.name else ""
        return (preset_tag in primary_tags
                and not c.is_sanity_only
                and not c.known_bad)

    def is_eligible(c: Candidate) -> bool:
        """Anything we'd consider letting into the champion comparison."""
        return not c.is_sanity_only and not c.known_bad

    # 2. Always re-add the champion's exact config.
    champ_fc = _champion_feature_config(prev_champion)
    champion_name = prev_champion["name"] if prev_champion else None
    if prev_champion and not prev_champion.get("is_baseline", False):
        match = next((c for c in full_space if c.name == champion_name), None)
        if match is not None:
            candidates.append(match)
        else:
            if champ_fc is not None:
                fc_hits = [c for c in full_space
                           if _fc_signature(c.feature_config) == _fc_signature(champ_fc)]
                if fc_hits:
                    candidates.append(fc_hits[0])

    if seed is None:
        seed = len(history) + 1
    rng = random.Random(seed)

    reserved = {c.name for c in candidates}

    def have_enough() -> bool:
        non_special = [x for x in candidates
                       if not x.is_baseline and x.name != champion_name]
        return len(non_special) >= n_challengers

    # 3. Champion mutations (primary first)
    if champ_fc is not None:
        mutations_primary = [
            c for c in full_space
            if c.name not in reserved
            and c.name not in history
            and is_eligible(c)
            and is_primary_candidate(c)
            and _knob_distance(c.feature_config, champ_fc) == 1
        ]
        mutations_other = [
            c for c in full_space
            if c.name not in reserved
            and c.name not in history
            and is_eligible(c)
            and not is_primary_candidate(c)
            and _knob_distance(c.feature_config, champ_fc) == 1
        ]
        rng.shuffle(mutations_primary)
        rng.shuffle(mutations_other)
        for c in (mutations_primary + mutations_other):
            if have_enough():
                break
            candidates.append(c)
            reserved.add(c.name)

    # 4. Primary untried random draw
    primary_untried = [
        c for c in full_space
        if c.name not in reserved
        and c.name not in history
        and is_eligible(c)
        and is_primary_candidate(c)
    ]
    rng.shuffle(primary_untried)
    for c in primary_untried:
        if have_enough():
            break
        candidates.append(c)
        reserved.add(c.name)

    # 5. Non-primary fallback
    other_untried = [
        c for c in full_space
        if c.name not in reserved
        and c.name not in history
        and is_eligible(c)
        and not is_primary_candidate(c)
    ]
    rng.shuffle(other_untried)
    for c in other_untried:
        if have_enough():
            break
        candidates.append(c)
        reserved.add(c.name)

    # 6. Last-resort variance check: include a tried point with a fresh seed
    if not have_enough():
        last_resort = [c for c in full_space
                       if c.name not in reserved and is_eligible(c)]
        rng.shuffle(last_resort)
        for c in last_resort:
            if have_enough():
                break
            candidates.append(c)
            reserved.add(c.name)

    return candidates


# ============================================================================
# Scoring -- holdout (iter-1) and walk-forward (iter-2)
# ============================================================================

@dataclass
class CandidateResult:
    name: str
    feature_config: Dict[str, Any]
    is_baseline: bool

    # ---- iter-1 holdout-style fields (still populated for backwards compat) -
    # In walk-forward mode these are aliases for the per-fold mean.
    metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    # ---- iter-2 walk-forward fields ----------------------------------------
    # Means and stds across folds (val side); std_indep is the std across the
    # non-overlapping subset (every-other fold, indices 0, 2, 4, ...).
    fold_metrics: Dict[str, float] = field(default_factory=dict)
    fold_test_metrics: Dict[str, float] = field(default_factory=dict)
    n_folds: int = 0
    per_fold_val: List[Dict[str, float]] = field(default_factory=list)
    per_fold_test: List[Dict[str, float]] = field(default_factory=list)

    # ---- bookkeeping -------------------------------------------------------
    runtime_sec: float = 0.0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_features: int = 0
    error: str | None = None
    uses_predicted_rrp: bool = False
    uses_observed_rrp: bool = False


def _is_baseline_model(model: Any) -> bool:
    return isinstance(model, SeasonalNaive)


def _score_one_holdout(
    candidate: Candidate,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    evaluate_on_test: bool,
) -> Tuple[Dict[str, float], Dict[str, float], int, int, int, int]:
    """Fit on `train`, score on `val` (always) and `test` (optional).

    Returns:
        (val_metrics, test_metrics, n_train, n_val, n_features, n_test)
    """
    model = candidate.model_factory()
    val_metrics: Dict[str, float] = {}
    test_metrics: Dict[str, float] = {}
    n_features = 0
    n_test_out = 0

    if _is_baseline_model(model):
        model.fit(train_df["time"], train_df["demand"].to_numpy())
        y_pred = model.predict(val_df["time"])
        y_true = val_df["demand"].to_numpy(dtype=float)
        val_metrics = score_all(y_true, y_pred)
        n_train_out = len(train_df)
        n_val_out = len(val_df)
        if evaluate_on_test and test_df is not None:
            y_pred_test = model.predict(test_df["time"])
            y_true_test = test_df["demand"].to_numpy(dtype=float)
            test_metrics = score_all(y_true_test, y_pred_test)
            n_test_out = len(test_df)
        return val_metrics, test_metrics, n_train_out, n_val_out, n_features, n_test_out

    fc = candidate.feature_config
    frames = [train_df, val_df]
    if evaluate_on_test and test_df is not None:
        frames.append(test_df)
    combined = pd.concat(frames).reset_index(drop=True)

    train_boundary = train_df["time"].max()
    val_boundary = val_df["time"].max()

    # ---- Two-stage RRP materialisation (fit stage-1 on train only) --------
    if getattr(fc, "use_predicted_rrp", False):
        from .predict_rrp import materialize_predicted_rrp
        train_mask_combined = combined["time"] <= train_boundary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence stage-1 sklearn chatter
            combined[PREDICTED_RRP_COL] = materialize_predicted_rrp(
                combined, train_mask_combined
            ).values

    X, y, feat_names, times = build_features(combined, fc)
    train_mask = times <= train_boundary
    val_mask = (times > train_boundary) & (times <= val_boundary)
    test_mask = times > val_boundary

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    val_metrics = score_all(y_val, y_pred)
    n_features = len(feat_names)
    n_train_out = int(train_mask.sum())
    n_val_out = int(val_mask.sum())
    if evaluate_on_test and test_df is not None:
        X_test, y_test = X[test_mask], y[test_mask]
        if len(y_test) > 0:
            y_pred_test = model.predict(X_test)
            test_metrics = score_all(y_test, y_pred_test)
            n_test_out = int(test_mask.sum())
    return val_metrics, test_metrics, n_train_out, n_val_out, n_features, n_test_out


def _agg_fold_metrics(per_fold: List[Dict[str, float]],
                      indep_indices: List[int]) -> Dict[str, float]:
    """Aggregate per-fold metric dicts to mean/std/std_indep per metric."""
    if not per_fold:
        return {}
    keys = list(per_fold[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([f[k] for f in per_fold], dtype=float)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if indep_indices and max(indep_indices) < len(vals):
            indep_vals = vals[indep_indices]
            out[f"{k}_std_indep"] = (
                float(np.std(indep_vals, ddof=1)) if len(indep_vals) > 1 else 0.0
            )
        else:
            out[f"{k}_std_indep"] = 0.0
    return out


def score_candidate(
    candidate: Candidate,
    splits_or_folds: Union[Splits, List[WalkForwardFold]],
    evaluate_on_test: bool = False,
) -> CandidateResult:
    """Fit + score a candidate.

    Dispatches on the second arg:
      * Splits                  -> single hold-out (iter-1 protocol)
      * list[WalkForwardFold]   -> walk-forward CV (iter-2 protocol)

    In walk-forward mode the returned CandidateResult.metrics carries the
    *mean* across folds (so existing iter-1 callers keep working), and
    `fold_metrics` adds the full mean/std/std_indep triples for each metric.
    """
    t0 = time.perf_counter()
    uses_predicted_rrp = bool(getattr(candidate.feature_config, "use_predicted_rrp", False))
    uses_observed_rrp = bool(getattr(candidate.feature_config, "use_rrp", False))

    try:
        # -------- Hold-out path (iter-1 protocol; unchanged semantics) ------
        if isinstance(splits_or_folds, Splits):
            splits = splits_or_folds
            val_metrics, test_metrics, n_train, n_val, n_features, n_test = (
                _score_one_holdout(
                    candidate, splits.train, splits.val,
                    splits.test if evaluate_on_test else None,
                    evaluate_on_test,
                )
            )
            runtime = time.perf_counter() - t0
            return CandidateResult(
                name=candidate.name,
                feature_config=asdict(candidate.feature_config),
                is_baseline=candidate.is_baseline,
                metrics=val_metrics,
                test_metrics=test_metrics,
                runtime_sec=runtime,
                n_train=n_train, n_val=n_val, n_test=n_test,
                n_features=n_features,
                uses_predicted_rrp=uses_predicted_rrp,
                uses_observed_rrp=uses_observed_rrp,
            )

        # -------- Walk-forward path (iter-2 protocol) -----------------------
        folds: List[WalkForwardFold] = splits_or_folds
        if not folds:
            raise ValueError("score_candidate received an empty folds list.")

        per_fold_val: List[Dict[str, float]] = []
        per_fold_test: List[Dict[str, float]] = []
        n_train_last = n_val_last = n_features_last = n_test_last = 0

        for fold in folds:
            val_m, test_m, ntr, nv, nf, nt = _score_one_holdout(
                candidate, fold.train, fold.val,
                fold.test if evaluate_on_test else None,
                evaluate_on_test,
            )
            per_fold_val.append(val_m)
            if evaluate_on_test and test_m:
                per_fold_test.append(test_m)
            n_train_last, n_val_last, n_features_last, n_test_last = ntr, nv, nf, nt

        indep_idx = list(range(0, len(folds), 2))   # 0, 2, 4, 6, 8 for 10 folds
        agg_val = _agg_fold_metrics(per_fold_val, indep_idx)
        agg_test = _agg_fold_metrics(per_fold_test, indep_idx) if per_fold_test else {}

        # `metrics` (iter-1 alias) = the mean column so downstream plots still
        # work. `fold_metrics` carries the full mean/std/std_indep triples.
        flat_val_mean = {
            PRIMARY_METRIC_NAME: agg_val.get(f"{PRIMARY_METRIC_NAME}_mean"),
            "rmse_demand": agg_val.get("rmse_demand_mean"),
            "mae_demand": agg_val.get("mae_demand_mean"),
        }
        flat_test_mean = {
            PRIMARY_METRIC_NAME: agg_test.get(f"{PRIMARY_METRIC_NAME}_mean"),
            "rmse_demand": agg_test.get("rmse_demand_mean"),
            "mae_demand": agg_test.get("mae_demand_mean"),
        } if agg_test else {}

        runtime = time.perf_counter() - t0
        return CandidateResult(
            name=candidate.name,
            feature_config=asdict(candidate.feature_config),
            is_baseline=candidate.is_baseline,
            metrics={k: v for k, v in flat_val_mean.items() if v is not None},
            test_metrics={k: v for k, v in flat_test_mean.items() if v is not None},
            fold_metrics=agg_val,
            fold_test_metrics=agg_test,
            n_folds=len(folds),
            per_fold_val=per_fold_val,
            per_fold_test=per_fold_test,
            runtime_sec=runtime,
            n_train=n_train_last, n_val=n_val_last,
            n_test=n_test_last, n_features=n_features_last,
            uses_predicted_rrp=uses_predicted_rrp,
            uses_observed_rrp=uses_observed_rrp,
        )

    except Exception as exc:
        runtime = time.perf_counter() - t0
        return CandidateResult(
            name=candidate.name,
            feature_config=asdict(candidate.feature_config),
            is_baseline=candidate.is_baseline,
            metrics={},
            runtime_sec=runtime,
            n_train=0, n_val=0, n_features=0,
            error=f"{type(exc).__name__}: {exc}",
            uses_predicted_rrp=uses_predicted_rrp,
            uses_observed_rrp=uses_observed_rrp,
        )


# ============================================================================
# Run report + persistence
# ============================================================================

@dataclass
class RunReport:
    run_id: str
    timestamp: str
    protocol: str                       # "holdout" | "walk_forward_v90"
    total_runtime_sec: float
    fit_count: int
    n_candidates: int
    splits_summary: str
    results: List[CandidateResult] = field(default_factory=list)
    best_name: str | None = None
    best_metric: float | None = None
    best_metric_std: float | None = None
    baseline_name: str | None = None
    baseline_metric: float | None = None
    baseline_metric_std: float | None = None
    improvement_vs_baseline: float | None = None
    prev_champion_name: str | None = None
    prev_champion_metric: float | None = None
    prev_champion_metric_std: float | None = None
    new_champion: bool = False
    improvement_vs_champion: float | None = None
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "protocol": self.protocol,
            "total_runtime_sec": self.total_runtime_sec,
            "fit_count": self.fit_count,
            "n_candidates": self.n_candidates,
            "splits_summary": self.splits_summary,
            "best_name": self.best_name,
            "best_metric": self.best_metric,
            "best_metric_std": self.best_metric_std,
            "baseline_name": self.baseline_name,
            "baseline_metric": self.baseline_metric,
            "baseline_metric_std": self.baseline_metric_std,
            "improvement_vs_baseline": self.improvement_vs_baseline,
            "prev_champion_name": self.prev_champion_name,
            "prev_champion_metric": self.prev_champion_metric,
            "prev_champion_metric_std": self.prev_champion_metric_std,
            "new_champion": self.new_champion,
            "improvement_vs_champion": self.improvement_vs_champion,
            "env": self.env,
            "results": [asdict(r) for r in self.results],
        }


MASTER_LOG_NAME = "master_log.csv"
CHAMPION_NAME = "champion.json"


def _load_champion(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_champion(path: Path, champion: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(champion, f, indent=2, default=str)


def _append_master_log(
    path: Path,
    run_id: str,
    timestamp: str,
    protocol: str,
    results: List[CandidateResult],
) -> None:
    """Append one row per candidate to the cross-run master log.

    Iter-2 columns (mean/std/std_indep per metric, n_folds, uses_*_rrp)
    are populated whenever the run is walk-forward; the iter-1
    `mse_demand` / `rmse_demand` / `mae_demand` columns are kept
    populated as aliases for the mean so downstream readers keep
    working unchanged.
    """
    rows = []
    for r in results:
        fm = r.fold_metrics or {}
        ftm = r.fold_test_metrics or {}
        row = {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "protocol": protocol,
            "candidate_name": r.name,
            "is_baseline": r.is_baseline,
            # iter-1 columns (means in walk-forward mode)
            PRIMARY_METRIC_NAME: r.metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand": r.metrics.get("rmse_demand"),
            "mae_demand": r.metrics.get("mae_demand"),
            # iter-1 test columns
            "mse_demand_test": r.test_metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand_test": r.test_metrics.get("rmse_demand"),
            "mae_demand_test": r.test_metrics.get("mae_demand"),
            # iter-2 walk-forward columns
            f"{PRIMARY_METRIC_NAME}_mean": fm.get(f"{PRIMARY_METRIC_NAME}_mean"),
            f"{PRIMARY_METRIC_NAME}_std":  fm.get(f"{PRIMARY_METRIC_NAME}_std"),
            f"{PRIMARY_METRIC_NAME}_std_indep": fm.get(f"{PRIMARY_METRIC_NAME}_std_indep"),
            "rmse_demand_mean": fm.get("rmse_demand_mean"),
            "rmse_demand_std":  fm.get("rmse_demand_std"),
            "mae_demand_mean":  fm.get("mae_demand_mean"),
            "mae_demand_std":   fm.get("mae_demand_std"),
            f"{PRIMARY_METRIC_NAME}_test_mean": ftm.get(f"{PRIMARY_METRIC_NAME}_mean"),
            f"{PRIMARY_METRIC_NAME}_test_std":  ftm.get(f"{PRIMARY_METRIC_NAME}_std"),
            "n_folds": r.n_folds,
            "uses_predicted_rrp": r.uses_predicted_rrp,
            "uses_observed_rrp": r.uses_observed_rrp,
            # bookkeeping
            "runtime_sec": r.runtime_sec,
            "n_features": r.n_features,
            "n_train": r.n_train,
            "n_val": r.n_val,
            "n_test": r.n_test,
            "error": r.error or "",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if path.exists():
        # Re-align columns with the existing file so the row order is stable
        # and old columns aren't accidentally dropped.
        existing = pd.read_csv(path, nrows=0)
        for col in df.columns:
            if col not in existing.columns:
                # New column: read the whole existing file and rewrite with
                # the new column added (NaN for old rows).
                old_df = pd.read_csv(path)
                for c in df.columns:
                    if c not in old_df.columns:
                        old_df[c] = np.nan
                pd.concat([old_df, df], ignore_index=True).to_csv(path, index=False)
                return
        df.to_csv(path, mode="a", header=False, index=False,
                  columns=list(existing.columns))
    else:
        df.to_csv(path, index=False)


# Promotion rule policies
PROMOTE_VAL = "val"
PROMOTE_TEST = "test"


def _noise_aware_promote(
    challenger_mean: float, challenger_std: float,
    champion_mean: float | None, champion_std: float | None,
) -> bool:
    """Iter-2 noise-aware promotion rule.

    Promote iff:  challenger_mean + challenger_std < champion_mean − champion_std

    With champion absent, any non-error result promotes (first run). With
    iter-1 deterministic point estimates (std=0), this collapses to the
    iter-1 strict-better rule.
    """
    if champion_mean is None:
        return True
    cs = challenger_std or 0.0
    ms = champion_std or 0.0
    return (challenger_mean + cs) < (champion_mean - ms)


def _result_score(r: CandidateResult, promote_on: str) -> float | None:
    """Mean metric for ranking under the chosen policy."""
    if promote_on == PROMOTE_TEST:
        return r.test_metrics.get(PRIMARY_METRIC_NAME)
    return r.metrics.get(PRIMARY_METRIC_NAME)


def _result_std(r: CandidateResult, promote_on: str) -> float:
    """Std (across folds) for the ranking metric under the chosen policy."""
    fm = (r.fold_test_metrics if promote_on == PROMOTE_TEST else r.fold_metrics) or {}
    return float(fm.get(f"{PRIMARY_METRIC_NAME}_std", 0.0) or 0.0)


def run_loop(
    splits_or_folds: Union[Splits, List[WalkForwardFold]],
    candidates: List[Candidate] | None = None,
    results_dir: Path | str = "experiments/auto_runs",
    evaluate_on_test: bool = False,
    promote_on: str = PROMOTE_VAL,
) -> RunReport:
    """Fit + score every candidate once; write artifacts; return report.

    Args:
        splits_or_folds: a `Splits` (iter-1 holdout protocol) or a
            list of `WalkForwardFold` (iter-2 protocol). The shape of
            this argument decides the protocol.
        evaluate_on_test: also score every candidate on the locked test
            set (the same train-only model is used for val and test).
        promote_on: 'val' (default) or 'test'.
    """
    candidates = candidates or default_candidates()
    if promote_on not in (PROMOTE_VAL, PROMOTE_TEST):
        raise ValueError(f"promote_on must be 'val' or 'test', got {promote_on!r}")
    if promote_on == PROMOTE_TEST and not evaluate_on_test:
        raise ValueError(
            "promote_on='test' requires evaluate_on_test=True; otherwise "
            "candidates have no test metric to rank by."
        )

    # Detect protocol from input shape.
    is_walk_forward = isinstance(splits_or_folds, list)
    protocol = "walk_forward_v90" if is_walk_forward else "holdout"
    if is_walk_forward:
        folds: List[WalkForwardFold] = splits_or_folds
        splits_summary = (
            f"walk-forward: {len(folds)} folds; "
            f"val window {len(folds[0].val)} days; "
            f"train expands {len(folds[0].train)} -> {len(folds[-1].train)} rows; "
            f"locked test: {folds[0].test['time'].min().date()} -> "
            f"{folds[0].test['time'].max().date()}"
        )
    else:
        splits: Splits = splits_or_folds
        splits_summary = splits.describe()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    master_log_path = results_dir / MASTER_LOG_NAME
    champion_path = results_dir / CHAMPION_NAME
    prev_champion = _load_champion(champion_path)

    if promote_on == PROMOTE_TEST:
        print("=" * 70)
        print("WARNING: promote_on='test' is enabled.")
        print("  The locked 365-day test set is being used as a tuning surface.")
        print("  See ERROR_TAXONOMY.md item L5. Use only when you understand")
        print("  the trade-off.")
        print("=" * 70)

    metric_label = f"{PRIMARY_METRIC_NAME}_test" if promote_on == PROMOTE_TEST else PRIMARY_METRIC_NAME

    if prev_champion:
        m = prev_champion.get("metric_mean") or prev_champion.get("metric")
        s = prev_champion.get("metric_std")
        msg = f"{metric_label}={m:.2f}"
        if s is not None:
            msg += f" ± {s:.2f}"
        print(f"[autoresearch] current champion: {prev_champion['name']} ({msg}, "
              f"from run {prev_champion.get('run_id', '?')}, "
              f"protocol={prev_champion.get('protocol', '?')})")
    else:
        print("[autoresearch] no champion on file yet -- this run will set one")

    t0 = time.perf_counter()
    results: List[CandidateResult] = []
    for cand in candidates:
        print(f"[autoresearch] scoring: {cand.describe()}")
        res = score_candidate(cand, splits_or_folds, evaluate_on_test=evaluate_on_test)
        if res.error:
            print(f"  ! error: {res.error}  ({res.runtime_sec:.2f}s)")
        else:
            val_mean = res.metrics.get(PRIMARY_METRIC_NAME)
            std = res.fold_metrics.get(f"{PRIMARY_METRIC_NAME}_std", 0.0)
            val_str = (f"val={val_mean:.2f} ± {std:.2f}" if is_walk_forward
                       else f"val={val_mean:.2f}")
            test_str = ""
            if evaluate_on_test and res.test_metrics:
                test_mean = res.test_metrics.get(PRIMARY_METRIC_NAME)
                test_str = f"  test={test_mean:.2f}"
            print(f"  {val_str}{test_str} ({res.runtime_sec:.2f}s, "
                  f"n_features={res.n_features}, n_folds={res.n_folds or 1})")
        results.append(res)
    total_runtime = time.perf_counter() - t0

    def _rank_key(r: CandidateResult) -> float:
        v = _result_score(r, promote_on)
        return v if v is not None else float("inf")

    ranked = sorted(
        [r for r in results if not r.error and _result_score(r, promote_on) is not None],
        key=_rank_key,
    )
    best = ranked[0] if ranked else None
    baseline_results = [r for r in results if r.is_baseline and not r.error
                        and _result_score(r, promote_on) is not None]
    best_baseline = min(baseline_results, key=_rank_key) if baseline_results else None
    baseline_metric = _result_score(best_baseline, promote_on) if best_baseline else None
    baseline_metric_std = _result_std(best_baseline, promote_on) if best_baseline else None
    improvement = (baseline_metric - _result_score(best, promote_on)
                   if best and baseline_metric is not None else None)

    new_champion_promoted = False
    improvement_vs_champion: float | None = None
    best_mean = _result_score(best, promote_on) if best else None
    best_std = _result_std(best, promote_on) if best else 0.0
    if best is not None:
        if prev_champion is None:
            new_champion_promoted = True
        else:
            prev_mean = prev_champion.get("metric_mean") or prev_champion.get("metric")
            prev_std = prev_champion.get("metric_std") or 0.0
            improvement_vs_champion = (prev_mean - best_mean) if prev_mean is not None else None
            new_champion_promoted = _noise_aware_promote(
                best_mean, best_std, prev_mean, prev_std
            )

    prev_metric = (prev_champion.get("metric_mean") if prev_champion
                   else None) or (prev_champion.get("metric") if prev_champion else None)
    prev_metric_std = prev_champion.get("metric_std") if prev_champion else None
    report = RunReport(
        run_id=str(uuid.uuid4())[:8],
        timestamp=pd.Timestamp.utcnow().isoformat(),
        protocol=protocol,
        total_runtime_sec=total_runtime,
        fit_count=sum(max(r.n_folds, 1) for r in results),
        n_candidates=len(candidates),
        splits_summary=splits_summary,
        results=results,
        best_name=best.name if best else None,
        best_metric=best_mean,
        best_metric_std=best_std,
        baseline_name=best_baseline.name if best_baseline else None,
        baseline_metric=baseline_metric,
        baseline_metric_std=baseline_metric_std,
        improvement_vs_baseline=improvement,
        prev_champion_name=prev_champion["name"] if prev_champion else None,
        prev_champion_metric=prev_metric,
        prev_champion_metric_std=prev_metric_std,
        new_champion=new_champion_promoted,
        improvement_vs_champion=improvement_vs_champion,
        env={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "promote_on": promote_on,
            "evaluate_on_test": str(evaluate_on_test),
            "protocol": protocol,
        },
    )

    # Run JSON
    out_json = results_dir / f"run_{report.run_id}.json"
    with open(out_json, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    # Leaderboard CSV
    leaderboard_rows = []
    for r in results:
        fm = r.fold_metrics or {}
        row = {
            "name": r.name,
            "is_baseline": r.is_baseline,
            "uses_predicted_rrp": r.uses_predicted_rrp,
            "uses_observed_rrp": r.uses_observed_rrp,
            f"{PRIMARY_METRIC_NAME}": r.metrics.get(PRIMARY_METRIC_NAME, np.nan),
            f"{PRIMARY_METRIC_NAME}_mean": fm.get(f"{PRIMARY_METRIC_NAME}_mean", np.nan),
            f"{PRIMARY_METRIC_NAME}_std":  fm.get(f"{PRIMARY_METRIC_NAME}_std", np.nan),
            f"{PRIMARY_METRIC_NAME}_std_indep": fm.get(f"{PRIMARY_METRIC_NAME}_std_indep", np.nan),
            "rmse_demand_mean": fm.get("rmse_demand_mean", np.nan),
            "mae_demand_mean": fm.get("mae_demand_mean", np.nan),
            "n_folds": r.n_folds,
        }
        if evaluate_on_test:
            row["mse_demand_test"] = r.test_metrics.get(PRIMARY_METRIC_NAME, np.nan)
        row.update({
            "runtime_sec": r.runtime_sec,
            "n_features": r.n_features,
            "error": r.error or "",
        })
        leaderboard_rows.append(row)
    sort_col = "mse_demand_test" if promote_on == PROMOTE_TEST else PRIMARY_METRIC_NAME
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(sort_col, na_position="last")
    leaderboard.to_csv(results_dir / f"run_{report.run_id}_leaderboard.csv", index=False)

    print()
    print(f"[autoresearch] run_id={report.run_id} total={total_runtime:.2f}s "
          f"budget={report.fit_count} fits  protocol={protocol}  promote_on={promote_on}")
    if best:
        std_str = f" ± {best_std:.2f}" if is_walk_forward else ""
        print(f"[autoresearch] best: {report.best_name}  "
              f"{metric_label}={report.best_metric:.2f}{std_str}")
    if improvement is not None:
        sign = "BELOW" if improvement > 0 else "ABOVE"
        print(f"[autoresearch] best baseline: {best_baseline.name} "
              f"({baseline_metric:.2f}) -> best is {abs(improvement):.2f} "
              f"{sign} best baseline")

    _append_master_log(master_log_path, report.run_id, report.timestamp, protocol, results)

    if best is not None and new_champion_promoted:
        # Iter-2 schema: persist mean + std + std_indep + n_folds + protocol.
        fm = best.fold_metrics or {}
        ftm = best.fold_test_metrics or {}
        champion_record = {
            "name": best.name,
            "metric": best_mean,                              # legacy alias
            "metric_source": promote_on,
            "metric_mean": best_mean,
            "metric_std": best_std,
            "metric_std_indep": fm.get(f"{PRIMARY_METRIC_NAME}_std_indep", 0.0),
            "n_folds": best.n_folds,
            "protocol": protocol,
            "val_metric": best.metrics.get(PRIMARY_METRIC_NAME),
            "test_metric": best.test_metrics.get(PRIMARY_METRIC_NAME)
                            if best.test_metrics else None,
            "test_metric_mean": ftm.get(f"{PRIMARY_METRIC_NAME}_mean") if ftm else None,
            "test_metric_std": ftm.get(f"{PRIMARY_METRIC_NAME}_std") if ftm else None,
            "uses_predicted_rrp": best.uses_predicted_rrp,
            "uses_observed_rrp": best.uses_observed_rrp,
            "run_id": report.run_id,
            "timestamp_utc": report.timestamp,
            "feature_config": best.feature_config,
            "is_baseline": best.is_baseline,
        }
        _save_champion(champion_path, champion_record)
        if prev_champion is None:
            print(f"[autoresearch] CHAMPION SET: {best.name} "
                  f"({metric_label}={best_mean:.2f} ± {best_std:.2f})")
        else:
            print(f"[autoresearch] NEW CHAMPION (noise-aware): {best.name} beats "
                  f"{prev_champion['name']} -- "
                  f"{prev_metric:.2f} ± {prev_metric_std or 0.0:.2f}  ->  "
                  f"{best_mean:.2f} ± {best_std:.2f}")
    elif prev_champion is not None and best is not None:
        prev_mean = prev_metric
        prev_std = prev_metric_std or 0.0
        why = (
            f"challenger {best_mean:.2f} ± {best_std:.2f} not strictly better "
            f"than {prev_mean:.2f} ± {prev_std:.2f} under noise-aware rule "
            f"(need challenger_mean + std < champion_mean − std)"
        )
        print(f"[autoresearch] champion unchanged: {prev_champion['name']} still leads; {why}")

    print(f"[autoresearch] wrote {out_json.name} + leaderboard.csv to {results_dir}")
    print(f"[autoresearch] appended {len(results)} rows to {MASTER_LOG_NAME}")

    return report
