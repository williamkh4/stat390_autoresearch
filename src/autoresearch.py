"""
AutoResearch loop.

One "run" of the loop = fit and score every registered candidate on the
validation split, write a leaderboard, and return the best candidate's config.
The locked 365-day test set is never touched here.

A candidate is a (feature_config, model_factory) pair. Swapping in new
candidates is how you "iterate" the research loop between runs.

Budget accounting:
  * runtime_sec per candidate and total wall-clock for the run
  * fit_count (number of model fits) as a simple "budget unit"
  * peak memory and gpu are out of scope for this first version
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Any, Tuple
import importlib
import itertools
import json
import platform
import random
import time
import uuid

import numpy as np
import pandas as pd

from .baselines import SeasonalNaive
from .features import FeatureConfig, build_features
from .metrics import PRIMARY_METRIC_NAME, score_all
from .numpy_models import NumpyOLS
from .split import Splits


# ----------------------------------------------------------------------------
# Candidate definitions
# ----------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    feature_config: FeatureConfig
    model_factory: Callable[[], Any]   # returns a fresh sklearn-like estimator
    is_baseline: bool = False

    def describe(self) -> str:
        return f"{self.name} | features[{self.feature_config.describe()}]"


def baseline_candidates() -> List[Candidate]:
    """Baseline reference models. Depend only on numpy/pandas, not sklearn.

    These are the "numbers to beat". run_baseline.py uses this list directly
    so the baseline analysis is usable even without sklearn installed.
    """
    empty_fc = FeatureConfig(
        use_calendar=False, use_temp=False, use_apparent_temp=False,
        use_rrp=False, demand_lags=[], rolling_windows=[],
    )
    return [
        # -- Baseline 1: weekly seasonal naive, period = 7 days.
        # Uses last week's observation; "floor" baseline every model must beat.
        Candidate(
            name="seasonal_naive_7",
            feature_config=empty_fc,
            model_factory=lambda: SeasonalNaive(period=7),
            is_baseline=True,
        ),
        # -- Baseline 2: yearly-aligned seasonal naive, period = 52*7 = 364.
        # Same day-of-week, 52 weeks ago; captures the annual cycle trivially.
        # A real model must beat this to justify using weather + RRP features.
        Candidate(
            name="seasonal_naive_364",
            feature_config=empty_fc,
            model_factory=lambda: SeasonalNaive(period=364),
            is_baseline=True,
        ),
    ]


def default_candidates() -> List[Candidate]:
    """Seed set for the first AutoResearch iteration = baselines + real models.

    Keep this short and cheap: the whole point of measuring runtime/budget
    is that you can grow this list deliberately, not by accident.
    """
    # Lazy imports: only required when the AutoResearch loop runs (not when
    # run_baseline.py runs), so baseline analysis works without sklearn.
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    return baseline_candidates() + [
        # -- Ridge: calendar + temperature only (classic feature set)
        Candidate(
            name="ridge_cal_temp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=False,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: Ridge(alpha=1.0, random_state=0),
        ),
        # -- Ridge: + apparent temperature (tests the core project hypothesis)
        Candidate(
            name="ridge_cal_temp_apptemp",
            feature_config=FeatureConfig(
                use_calendar=True, use_temp=True, use_apparent_temp=True,
                use_rrp=False, demand_lags=[1, 7], rolling_windows=[7],
            ),
            model_factory=lambda: Ridge(alpha=1.0, random_state=0),
        ),
        # -- Random Forest: same features as above
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
        # -- GBM: full feature stack including RRP (observed, not predicted)
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


# ----------------------------------------------------------------------------
# Auto-generated search space
# ----------------------------------------------------------------------------
#
# The goal: every `python run_autoresearch.py` invocation should propose new
# work without manual editing. The strategy is two-fold:
#
#   1. There is a *fixed* discrete search space of (model_spec x feature_preset).
#      Each pair has a deterministic readable name (so two runs that pick the
#      same combo log it under the same row).
#   2. Each run reads `master_log.csv` for everything that's already been tried,
#      then picks N challengers preferring (a) one-knob mutations of the
#      *current champion's* feature_config, then (b) untried random points from
#      the broader space, then falls back to (c) lightly tweaking already-tried
#      candidates if the space has been exhausted.
#
# Baselines + the champion's exact config are always re-run so each run's
# leaderboard is interpretable on its own without joining against history.

# --- Model specs --------------------------------------------------------------
# A "model spec" is (model_type_str, kwargs_tuple). Tuples (not dicts) so they
# hash and dedupe cleanly. _make_factory turns a spec into a model_factory.
#
# `numpy_ols` works with no extra dependencies. The sklearn entries are
# auto-skipped at runtime if sklearn isn't importable.
#
# Each block below is curated -- not Cartesian -- so the search space stays
# interpretable. To enlarge it, append more tuples; the auto-generator will
# pick them up automatically the next time `python run_autoresearch.py` runs.
MODEL_SPECS: List[Tuple[str, Tuple[Tuple[str, Any], ...]]] = [
    # --- Linear: numpy-only OLS / ridge (no sklearn dep) ----------------------
    ("numpy_ols", (("alpha", 0.0),)),
    ("numpy_ols", (("alpha", 1.0),)),
    ("numpy_ols", (("alpha", 10.0),)),

    # --- Linear: sklearn Ridge over a wider alpha sweep -----------------------
    ("ridge",     (("alpha", 0.01), ("random_state", 0))),
    ("ridge",     (("alpha", 0.1),  ("random_state", 0))),
    ("ridge",     (("alpha", 1.0),  ("random_state", 0))),
    ("ridge",     (("alpha", 10.0), ("random_state", 0))),
    ("ridge",     (("alpha", 100.0),("random_state", 0))),

    # --- Random Forest --------------------------------------------------------
    # Knobs varied: n_estimators, max_depth, min_samples_leaf, max_features,
    # min_samples_split, bootstrap. Defaults filled in for the rest.
    ("rf", (("n_estimators", 100), ("max_depth", None), ("min_samples_leaf", 1),
            ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 1),
            ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", 10),   ("min_samples_leaf", 1),
            ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 5),
            ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 10),
            ("max_features", "sqrt"), ("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 1),
            ("max_features", 1.0),   ("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 1),
            ("max_features", "log2"),("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 500), ("max_depth", None), ("min_samples_leaf", 5),
            ("max_features", "sqrt"),("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 5),
            ("max_features", "sqrt"),("bootstrap", False),
            ("random_state", 0), ("n_jobs", -1))),
    ("rf", (("n_estimators", 200), ("max_depth", None), ("min_samples_leaf", 5),
            ("min_samples_split", 10), ("max_features", "sqrt"),
            ("random_state", 0), ("n_jobs", -1))),

    # --- Gradient Boosting ----------------------------------------------------
    # Knobs varied: n_estimators, max_depth, learning_rate, subsample,
    # min_samples_leaf, max_features, loss, early stopping.
    ("gbm", (("n_estimators", 200), ("max_depth", 3), ("learning_rate", 0.1),
             ("subsample", 1.0), ("random_state", 0))),
    ("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
             ("subsample", 1.0), ("random_state", 0))),
    ("gbm", (("n_estimators", 300), ("max_depth", 5), ("learning_rate", 0.05),
             ("subsample", 1.0), ("random_state", 0))),
    ("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
             ("subsample", 0.8), ("random_state", 0))),
    ("gbm", (("n_estimators", 500), ("max_depth", 3), ("learning_rate", 0.03),
             ("subsample", 0.8), ("random_state", 0))),
    ("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
             ("subsample", 1.0), ("min_samples_leaf", 5), ("random_state", 0))),
    ("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
             ("subsample", 1.0), ("min_samples_leaf", 10), ("random_state", 0))),
    ("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
             ("subsample", 1.0), ("max_features", "sqrt"), ("random_state", 0))),
    ("gbm", (("n_estimators", 300), ("max_depth", 3), ("learning_rate", 0.05),
             ("subsample", 1.0), ("loss", "huber"), ("random_state", 0))),
    ("gbm", (("n_estimators", 500), ("max_depth", 3), ("learning_rate", 0.05),
             ("subsample", 0.8), ("n_iter_no_change", 10),
             ("validation_fraction", 0.15), ("random_state", 0))),

    # --- Multi-layer perceptron (MLPRegressor in a StandardScaler pipeline) --
    # Knobs varied: hidden_layer_sizes, activation, solver, alpha (L2),
    # learning_rate_init, max_iter, early_stopping. Inputs are scaled in the
    # pipeline because MLP convergence is very scale-sensitive.
    ("mlp", (("hidden_layer_sizes", (64,)),         ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.0001), ("learning_rate_init", 0.001),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (64, 32)),      ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.0001), ("learning_rate_init", 0.001),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (128, 64, 32)), ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.0001), ("learning_rate_init", 0.001),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (64, 32)),      ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.001),  ("learning_rate_init", 0.001),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (64, 32)),      ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.01),   ("learning_rate_init", 0.001),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (64, 32)),      ("activation", "tanh"), ("solver", "adam"),
             ("alpha", 0.0001), ("learning_rate_init", 0.001),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (64, 32)),      ("activation", "relu"), ("solver", "lbfgs"),
             ("alpha", 0.0001), ("max_iter", 1000), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (32, 16)),      ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.001),  ("learning_rate_init", 0.001),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (128, 64)),     ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.001),  ("learning_rate_init", 0.01),
             ("max_iter", 500), ("random_state", 0))),
    ("mlp", (("hidden_layer_sizes", (64, 32)),      ("activation", "relu"), ("solver", "adam"),
             ("alpha", 0.001),  ("learning_rate_init", 0.001),
             ("max_iter", 1000), ("early_stopping", True),
             ("validation_fraction", 0.15), ("random_state", 0))),
]

# --- Feature presets ----------------------------------------------------------
# Hand-curated points in feature space. Each one has a short tag for the
# auto-generated candidate name. New tags should be added to FEATURE_PRESET_TAGS
# below.
def _fc(use_calendar=False, use_temp=False, use_apparent_temp=False,
        use_rrp=False, demand_lags=None, rolling_windows=None) -> FeatureConfig:
    return FeatureConfig(
        use_calendar=use_calendar, use_temp=use_temp,
        use_apparent_temp=use_apparent_temp, use_rrp=use_rrp,
        demand_lags=list(demand_lags or []),
        rolling_windows=list(rolling_windows or []),
    )


FEATURE_PRESETS: List[Tuple[str, FeatureConfig]] = [
    ("cal",                 _fc(use_calendar=True)),
    ("cal_lag1-7",          _fc(use_calendar=True, demand_lags=[1, 7])),
    ("cal_lag1-7_roll7",    _fc(use_calendar=True, demand_lags=[1, 7], rolling_windows=[7])),
    ("cal_lag1-7-14_roll7-28",
                            _fc(use_calendar=True, demand_lags=[1, 7, 14], rolling_windows=[7, 28])),
    ("cal_temp",            _fc(use_calendar=True, use_temp=True)),
    ("cal_temp_lag1-7",     _fc(use_calendar=True, use_temp=True, demand_lags=[1, 7])),
    ("cal_temp_lag1-7_roll7",
                            _fc(use_calendar=True, use_temp=True, demand_lags=[1, 7], rolling_windows=[7])),
    ("cal_temp_apptemp",    _fc(use_calendar=True, use_temp=True, use_apparent_temp=True)),
    ("cal_temp_apptemp_lag1-7",
                            _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                                demand_lags=[1, 7])),
    ("cal_temp_apptemp_lag1-7_roll7",
                            _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                                demand_lags=[1, 7], rolling_windows=[7])),
    ("cal_temp_apptemp_lag1-7-14_roll7-28",
                            _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                                demand_lags=[1, 7, 14], rolling_windows=[7, 28])),
    ("full",                _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                                use_rrp=True, demand_lags=[1, 7], rolling_windows=[7])),
    ("full_lag1-7-14_roll7-28",
                            _fc(use_calendar=True, use_temp=True, use_apparent_temp=True,
                                use_rrp=True, demand_lags=[1, 7, 14], rolling_windows=[7, 28])),
]


def _sklearn_available() -> bool:
    try:
        importlib.import_module("sklearn")
        return True
    except ImportError:
        return False


def _model_spec_compatible(model_type: str) -> bool:
    """Whether the host has the libs needed for this model spec."""
    if model_type == "numpy_ols":
        return True
    if model_type in {"ridge", "rf", "gbm", "mlp"}:
        return _sklearn_available()
    return False


def _format_kwargs(kwargs: Tuple[Tuple[str, Any], ...]) -> str:
    """Stable suffix for the candidate name, e.g. 'alpha1.0_n200'."""
    parts: List[str] = []
    # Short tokens for readability. Anything not in this dict uses its full key.
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
    # Knobs that are reproducibility / performance overrides, not search axes.
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
    """Deterministic readable name. Same (spec, preset) -> same name across runs."""
    suffix = _format_kwargs(model_kwargs)
    return f"{model_type}__{preset_tag}__{suffix}" if suffix else f"{model_type}__{preset_tag}"


def _make_factory(model_type: str,
                  model_kwargs: Tuple[Tuple[str, Any], ...]) -> Callable[[], Any]:
    """Build a zero-arg factory for a given (model_type, kwargs) spec.

    sklearn imports are inside the factory so importing this module never
    requires sklearn -- baseline-only paths still work without it.
    """
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
        # MLP is wrapped in a StandardScaler so heterogeneous feature scales
        # don't dominate convergence. The pipeline implements the same
        # fit/predict interface so it slots into score_candidate unchanged.
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
    """All (model_spec x feature_preset) candidates compatible with the host."""
    out: List[Candidate] = []
    for (model_type, kwargs) in MODEL_SPECS:
        if not _model_spec_compatible(model_type):
            continue
        for (preset_tag, fc) in FEATURE_PRESETS:
            name = _config_name(model_type, kwargs, preset_tag)
            out.append(Candidate(
                name=name,
                feature_config=fc,
                model_factory=_make_factory(model_type, kwargs),
                is_baseline=False,
            ))
    return out


def _read_history_names(results_dir: Path) -> set[str]:
    """Names of every candidate that has ever been recorded in master_log.csv."""
    log_path = Path(results_dir) / MASTER_LOG_NAME
    if not log_path.exists():
        return set()
    try:
        df = pd.read_csv(log_path, usecols=["candidate_name"])
    except Exception:
        return set()
    return set(df["candidate_name"].dropna().astype(str).tolist())


def _fc_signature(fc: FeatureConfig) -> Tuple:
    """Hashable representation of a FeatureConfig, for one-knob comparison."""
    return (
        fc.use_calendar, fc.use_temp, fc.use_apparent_temp, fc.use_rrp,
        tuple(fc.demand_lags), tuple(fc.rolling_windows),
    )


def _knob_distance(fc_a: FeatureConfig, fc_b: FeatureConfig) -> int:
    """Number of feature-config knobs that differ between two configs."""
    sig_a = _fc_signature(fc_a)
    sig_b = _fc_signature(fc_b)
    return sum(1 for x, y in zip(sig_a, sig_b) if x != y)


def _champion_feature_config(champion: Dict[str, Any] | None) -> FeatureConfig | None:
    """Reconstruct a FeatureConfig from the champion record, if present."""
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
        demand_lags=list(fc.get("demand_lags", []) or []),
        rolling_windows=list(fc.get("rolling_windows", []) or []),
    )


def auto_candidates(
    prev_champion: Dict[str, Any] | None,
    results_dir: Path | str,
    n_challengers: int = 4,
    seed: int | None = None,
) -> List[Candidate]:
    """Build a self-driving candidate list for one AutoResearch iteration.

    Strategy (in priority order):

      1. **Baselines** -- always included. Cheap, and they keep every run's
         leaderboard interpretable in isolation.
      2. **Champion config re-run** -- always included so we can see noise on
         the current best and not lose it from the leaderboard.
      3. **One-knob mutations of the champion** -- candidates whose feature
         config differs from the champion's by exactly one switch/list. These
         are the cleanest ablations.
      4. **Random untried points** from the full search space. If the space
         is fully exhausted, we re-include a tried point with a fresh seed
         (variance check); if that's also undesired, we just include fewer.

    History dedupe is by candidate *name* (which is deterministic). All names
    already in `master_log.csv` are skipped from steps 3 and 4.

    Args:
        prev_champion: dict loaded from `champion.json`, or None for first run.
        results_dir: where master_log.csv lives.
        n_challengers: number of non-baseline, non-champion candidates to add.
        seed: optional rng seed; if None, derived from history length so each
            successive run picks new combos automatically.
    """
    candidates: List[Candidate] = list(baseline_candidates())
    history = _read_history_names(Path(results_dir))
    full_space = _full_search_space()

    # Always include the champion's exact config as the leader of the pack.
    # If the champion itself was a baseline, skip this step (already added).
    champ_fc = _champion_feature_config(prev_champion)
    champion_name = prev_champion["name"] if prev_champion else None
    if prev_champion and not prev_champion.get("is_baseline", False):
        match = next((c for c in full_space if c.name == champion_name), None)
        if match is not None:
            candidates.append(match)
        else:
            # Champion was registered under a non-auto name (e.g. legacy default
            # candidates). We can still re-add a Candidate by reconstructing it
            # from the search-space matching feature config; if multiple, pick
            # the first numpy_ols variant -- we don't know the original model.
            if champ_fc is not None:
                fc_hits = [c for c in full_space
                           if _fc_signature(c.feature_config) == _fc_signature(champ_fc)]
                if fc_hits:
                    candidates.append(fc_hits[0])

    # Random source: deterministic per history-length, so each run draws from a
    # fresh slice of the space without the user picking a seed.
    if seed is None:
        seed = len(history) + 1
    rng = random.Random(seed)

    # Pool of names already reserved this run (baselines + champion).
    reserved = {c.name for c in candidates}

    # 3. Champion mutations.
    if champ_fc is not None:
        mutations = [
            c for c in full_space
            if c.name not in reserved
            and c.name not in history
            and _knob_distance(c.feature_config, champ_fc) == 1
        ]
        rng.shuffle(mutations)
        for c in mutations:
            if len([x for x in candidates if not x.is_baseline and x.name != champion_name]) >= n_challengers:
                break
            candidates.append(c)
            reserved.add(c.name)

    # 4. Random untried points.
    untried = [c for c in full_space if c.name not in reserved and c.name not in history]
    rng.shuffle(untried)
    for c in untried:
        if len([x for x in candidates if not x.is_baseline and x.name != champion_name]) >= n_challengers:
            break
        candidates.append(c)
        reserved.add(c.name)

    # 5. Fallback: if we've exhausted the untried pool, sample tried points so
    # the user still gets a non-degenerate run (variance/confirmation).
    if len([x for x in candidates if not x.is_baseline and x.name != champion_name]) < n_challengers:
        fallback = [c for c in full_space if c.name not in reserved]
        rng.shuffle(fallback)
        for c in fallback:
            if len([x for x in candidates if not x.is_baseline and x.name != champion_name]) >= n_challengers:
                break
            candidates.append(c)
            reserved.add(c.name)

    return candidates


# ----------------------------------------------------------------------------
# Scoring one candidate
# ----------------------------------------------------------------------------

@dataclass
class CandidateResult:
    name: str
    feature_config: Dict[str, Any]
    is_baseline: bool
    metrics: Dict[str, float]            # validation metrics (primary surface)
    runtime_sec: float
    n_train: int
    n_val: int
    n_features: int
    error: str | None = None
    # Optional: only populated when run_loop is invoked with
    # evaluate_on_test=True. Kept separate from `metrics` so the validation
    # surface remains the canonical interpretation of "metrics" everywhere
    # else in the codebase, and so reading old run_<id>.json files (which
    # don't have this field) keeps working.
    test_metrics: Dict[str, float] = field(default_factory=dict)
    n_test: int = 0


def _is_baseline_model(model: Any) -> bool:
    return isinstance(model, SeasonalNaive)


def score_candidate(candidate: Candidate, splits: Splits,
                    evaluate_on_test: bool = False) -> CandidateResult:
    """Fit a candidate on train, score on val, optionally also on test.

    The validation score is the primary research surface and is always
    populated. When `evaluate_on_test=True`, the *same* train-only model
    additionally predicts on `splits.test` and the test metrics are
    populated. The model is NOT refit on train+val for this; that
    apples-to-apples comparison with val is more useful in-loop than the
    deployment-style refit (which `run_test_evaluation.py` provides for
    the final champion).
    """
    t0 = time.perf_counter()
    try:
        model = candidate.model_factory()
        test_metrics: Dict[str, float] = {}
        n_test = 0

        if _is_baseline_model(model):
            # Seasonal naive uses time + y directly.
            train_hist = pd.concat([splits.train, splits.val.iloc[:0]])  # train only
            model.fit(train_hist["time"], train_hist["demand"].to_numpy())
            y_pred = model.predict(splits.val["time"])
            y_true = splits.val["demand"].to_numpy(dtype=float)
            n_features = 0
            n_train = len(train_hist)
            n_val = len(splits.val)
            if evaluate_on_test:
                y_pred_test = model.predict(splits.test["time"])
                y_true_test = splits.test["demand"].to_numpy(dtype=float)
                test_metrics = score_all(y_true_test, y_pred_test)
                n_test = len(splits.test)
        else:
            fc = candidate.feature_config

            # Fit feature frame on the concatenation of train + val (and test
            # too, when needed) so lag/rolling windows are contiguous across
            # all split boundaries. The model still only sees train rows
            # during fit; val + test rows are predict-only.
            frames = [splits.train, splits.val]
            if evaluate_on_test:
                frames.append(splits.test)
            combined = pd.concat(frames).reset_index(drop=True)
            train_boundary = splits.train["time"].max()
            val_boundary = splits.val["time"].max()

            X, y, feat_names, times = build_features(combined, fc)
            train_mask = times <= train_boundary
            val_mask = (times > train_boundary) & (times <= val_boundary)
            test_mask = times > val_boundary

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_true = y_val
            n_features = len(feat_names)
            n_train = int(train_mask.sum())
            n_val = int(val_mask.sum())
            if evaluate_on_test:
                X_test, y_test = X[test_mask], y[test_mask]
                if len(y_test) > 0:
                    y_pred_test = model.predict(X_test)
                    test_metrics = score_all(y_test, y_pred_test)
                    n_test = int(test_mask.sum())

        metrics = score_all(y_true, y_pred)
        runtime = time.perf_counter() - t0
        return CandidateResult(
            name=candidate.name,
            feature_config=asdict(candidate.feature_config),
            is_baseline=candidate.is_baseline,
            metrics=metrics,
            test_metrics=test_metrics,
            runtime_sec=runtime,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_features=n_features,
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
        )


# ----------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------

@dataclass
class RunReport:
    run_id: str
    timestamp: str
    total_runtime_sec: float
    fit_count: int              # the "budget" spent this run
    n_candidates: int
    splits_summary: str
    results: List[CandidateResult] = field(default_factory=list)
    best_name: str | None = None
    best_metric: float | None = None
    baseline_name: str | None = None
    baseline_metric: float | None = None
    improvement_vs_baseline: float | None = None
    prev_champion_name: str | None = None
    prev_champion_metric: float | None = None
    new_champion: bool = False                      # did this run promote a new champion
    improvement_vs_champion: float | None = None
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_runtime_sec": self.total_runtime_sec,
            "fit_count": self.fit_count,
            "n_candidates": self.n_candidates,
            "splits_summary": self.splits_summary,
            "best_name": self.best_name,
            "best_metric": self.best_metric,
            "baseline_name": self.baseline_name,
            "baseline_metric": self.baseline_metric,
            "improvement_vs_baseline": self.improvement_vs_baseline,
            "prev_champion_name": self.prev_champion_name,
            "prev_champion_metric": self.prev_champion_metric,
            "new_champion": self.new_champion,
            "improvement_vs_champion": self.improvement_vs_champion,
            "env": self.env,
            "results": [asdict(r) for r in self.results],
        }


# ----------------------------------------------------------------------------
# Cross-run persistence: master log + champion
# ----------------------------------------------------------------------------

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
    results: List[CandidateResult],
) -> None:
    """Append one row per candidate to the cross-run master log.

    Test columns (`mse_demand_test`, etc.) are populated only when the
    run was invoked with `evaluate_on_test=True`; otherwise they are NaN
    and existing analyses keying off the val columns continue to work.
    """
    rows = [
        {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "candidate_name": r.name,
            "is_baseline": r.is_baseline,
            PRIMARY_METRIC_NAME: r.metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand": r.metrics.get("rmse_demand"),
            "mae_demand": r.metrics.get("mae_demand"),
            "mse_demand_test": r.test_metrics.get(PRIMARY_METRIC_NAME),
            "rmse_demand_test": r.test_metrics.get("rmse_demand"),
            "mae_demand_test": r.test_metrics.get("mae_demand"),
            "runtime_sec": r.runtime_sec,
            "n_features": r.n_features,
            "n_train": r.n_train,
            "n_val": r.n_val,
            "n_test": r.n_test,
            "error": r.error or "",
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# Where to read the candidate's score from when ranking / promoting champion.
PROMOTE_VAL = "val"
PROMOTE_TEST = "test"


def _result_score(r: CandidateResult, promote_on: str) -> float | None:
    """Return the metric value used for ranking under the chosen policy."""
    if promote_on == PROMOTE_TEST:
        return r.test_metrics.get(PRIMARY_METRIC_NAME)
    return r.metrics.get(PRIMARY_METRIC_NAME)


def run_loop(
    splits: Splits,
    candidates: List[Candidate] | None = None,
    results_dir: Path | str = "experiments/auto_runs",
    evaluate_on_test: bool = False,
    promote_on: str = PROMOTE_VAL,
) -> RunReport:
    """Fit + score every candidate once; write leaderboard; return report.

    Args:
        evaluate_on_test: If True, also score every candidate on the
            locked test set (the same train-only model is used to predict
            both val and test). Adds `*_test` columns to master_log and
            the run JSON. Default False keeps the locked test set
            untouched, preserving the held-out semantics.
        promote_on: One of ``"val"`` (default) or ``"test"``. Picks
            which metric source ranks candidates and decides champion
            promotion. ``"test"`` requires `evaluate_on_test=True` and
            burns the test set as a tuning surface -- the loop will
            print a warning when this is selected.
    """
    candidates = candidates or default_candidates()
    if promote_on not in (PROMOTE_VAL, PROMOTE_TEST):
        raise ValueError(f"promote_on must be 'val' or 'test', got {promote_on!r}")
    if promote_on == PROMOTE_TEST and not evaluate_on_test:
        raise ValueError(
            "promote_on='test' requires evaluate_on_test=True; otherwise "
            "candidates have no test metric to rank by."
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    master_log_path = results_dir / MASTER_LOG_NAME
    champion_path = results_dir / CHAMPION_NAME
    prev_champion = _load_champion(champion_path)

    if promote_on == PROMOTE_TEST:
        print("=" * 70)
        print("WARNING: promote_on='test' is enabled.")
        print("  The locked 365-day test set is being used as a tuning surface.")
        print("  Every candidate evaluated this way is one more 'look' at the")
        print("  held-out window; after enough runs, lowest-test-MSE wins by")
        print("  overfitting to it, not by generalising. See ERROR_TAXONOMY.md")
        print("  item L5 / FAILURE_ANALYSIS_MEMO.docx. Use this only when you")
        print("  understand the trade-off.")
        print("=" * 70)

    metric_label = f"{PRIMARY_METRIC_NAME}_test" if promote_on == PROMOTE_TEST \
        else PRIMARY_METRIC_NAME

    if prev_champion:
        print(f"[autoresearch] current champion: {prev_champion['name']} "
              f"({metric_label}={prev_champion['metric']:.2f}, "
              f"from run {prev_champion.get('run_id', '?')})")
    else:
        print("[autoresearch] no champion on file yet -- this run will set one")

    t0 = time.perf_counter()
    results: List[CandidateResult] = []
    for cand in candidates:
        print(f"[autoresearch] scoring: {cand.describe()}")
        res = score_candidate(cand, splits, evaluate_on_test=evaluate_on_test)
        if res.error:
            print(f"  ! error: {res.error}  ({res.runtime_sec:.2f}s)")
        else:
            val_str = f"val={res.metrics[PRIMARY_METRIC_NAME]:.2f}"
            test_str = (f"  test={res.test_metrics.get(PRIMARY_METRIC_NAME):.2f}"
                        if evaluate_on_test and res.test_metrics else "")
            print(f"  {val_str}{test_str} "
                  f"({res.runtime_sec:.2f}s, n_features={res.n_features})")
        results.append(res)
    total_runtime = time.perf_counter() - t0

    # Build leaderboard: best (lowest) primary metric first, ignoring errors.
    # Ranking metric depends on the promote_on policy.
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
    best_baseline = (
        min(baseline_results, key=_rank_key) if baseline_results else None
    )
    baseline_metric = _result_score(best_baseline, promote_on) if best_baseline else None
    improvement = None
    if best and baseline_metric is not None:
        improvement = baseline_metric - _result_score(best, promote_on)

    # Champion logic: promote best of this run if it beats the prior champion.
    new_champion_promoted = False
    improvement_vs_champion: float | None = None
    if best is not None:
        best_metric_val = _result_score(best, promote_on)
        if prev_champion is None:
            new_champion_promoted = True
        elif best_metric_val < prev_champion["metric"]:
            improvement_vs_champion = prev_champion["metric"] - best_metric_val
            new_champion_promoted = True
        else:
            improvement_vs_champion = prev_champion["metric"] - best_metric_val  # negative

    best_score = _result_score(best, promote_on) if best else None
    report = RunReport(
        run_id=str(uuid.uuid4())[:8],
        timestamp=pd.Timestamp.utcnow().isoformat(),
        total_runtime_sec=total_runtime,
        fit_count=len(results),
        n_candidates=len(candidates),
        splits_summary=splits.describe(),
        results=results,
        best_name=best.name if best else None,
        best_metric=best_score,
        baseline_name=best_baseline.name if best_baseline else None,
        baseline_metric=baseline_metric,
        improvement_vs_baseline=improvement,
        prev_champion_name=prev_champion["name"] if prev_champion else None,
        prev_champion_metric=prev_champion["metric"] if prev_champion else None,
        new_champion=new_champion_promoted,
        improvement_vs_champion=improvement_vs_champion,
        env={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "promote_on": promote_on,
            "evaluate_on_test": str(evaluate_on_test),
        },
    )

    # Write the run report as JSON + a compact leaderboard CSV.
    out_json = results_dir / f"run_{report.run_id}.json"
    with open(out_json, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    leaderboard_rows = []
    for r in results:
        row = {
            "name": r.name,
            "is_baseline": r.is_baseline,
            PRIMARY_METRIC_NAME: r.metrics.get(PRIMARY_METRIC_NAME, np.nan),
            "rmse_demand": r.metrics.get("rmse_demand", np.nan),
            "mae_demand": r.metrics.get("mae_demand", np.nan),
        }
        if evaluate_on_test:
            row["mse_demand_test"] = r.test_metrics.get(PRIMARY_METRIC_NAME, np.nan)
            row["rmse_demand_test"] = r.test_metrics.get("rmse_demand", np.nan)
            row["mae_demand_test"] = r.test_metrics.get("mae_demand", np.nan)
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
          f"budget={report.fit_count} fits  promote_on={promote_on}")
    if best:
        print(f"[autoresearch] best: {report.best_name}  "
              f"{metric_label}={report.best_metric:.2f}")
        # When test eval is on, also surface the val MSE so the gap is visible
        # even though it's not the promotion criterion.
        if evaluate_on_test and best.metrics and best.test_metrics:
            other_label = (PRIMARY_METRIC_NAME if promote_on == PROMOTE_TEST
                           else f"{PRIMARY_METRIC_NAME}_test")
            other_val = (best.metrics.get(PRIMARY_METRIC_NAME)
                         if promote_on == PROMOTE_TEST
                         else best.test_metrics.get(PRIMARY_METRIC_NAME))
            if other_val is not None:
                print(f"[autoresearch] best (other side): {other_label}={other_val:.2f}")
    if improvement is not None:
        sign = "BELOW" if improvement > 0 else "ABOVE"
        print(f"[autoresearch] best baseline: {best_baseline.name} "
              f"({baseline_metric:.2f}) -> best is {abs(improvement):.2f} "
              f"{sign} best baseline")

    # Champion bookkeeping: append every result to the master log; if this
    # run promoted a new champion, overwrite champion.json so future runs use
    # it as the bar to beat.
    _append_master_log(master_log_path, report.run_id, report.timestamp, results)

    if best is not None and new_champion_promoted:
        champion_record = {
            "name": best.name,
            "metric": best_score,
            "metric_source": promote_on,   # 'val' or 'test'; new field
            "val_metric": best.metrics.get(PRIMARY_METRIC_NAME),
            "test_metric": best.test_metrics.get(PRIMARY_METRIC_NAME) if best.test_metrics else None,
            "run_id": report.run_id,
            "timestamp_utc": report.timestamp,
            "feature_config": best.feature_config,
            "is_baseline": best.is_baseline,
        }
        _save_champion(champion_path, champion_record)
        if prev_champion is None:
            print(f"[autoresearch] CHAMPION SET: {best.name} "
                  f"({metric_label}={best_score:.2f})")
        else:
            print(f"[autoresearch] NEW CHAMPION: {best.name} beats "
                  f"{prev_champion['name']} by {improvement_vs_champion:.2f} "
                  f"({metric_label}: {prev_champion['metric']:.2f} -> {best_score:.2f})")
    elif prev_champion is not None and improvement_vs_champion is not None:
        print(f"[autoresearch] champion unchanged: {prev_champion['name']} "
              f"still leads ({metric_label}={prev_champion['metric']:.2f}); "
              f"this run's best was {abs(improvement_vs_champion):.2f} above it")

    print(f"[autoresearch] wrote {out_json.name} + leaderboard.csv to {results_dir}")
    print(f"[autoresearch] appended {len(results)} rows to {MASTER_LOG_NAME}")

    return report
