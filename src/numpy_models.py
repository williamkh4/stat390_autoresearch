"""
NumPy-only linear models for the AutoResearch loop.

These exist so the framework can run experiments without scikit-learn
installed (useful for sandboxed dry runs and for environments where you
want minimal dependencies). They expose the standard `fit(X, y) / predict(X)`
interface so they're interchangeable with sklearn estimators in the
AutoResearch candidate registry.

When sklearn is available, the sklearn versions in `default_candidates()`
are preferred -- they're more battle-tested and slightly faster on large
n. These NumPy versions are mathematically equivalent for the small
n we have here.
"""

from __future__ import annotations

import numpy as np


class NumpyOLS:
    """Ordinary least squares with optional L2 (ridge) regularization.

    Features are standardized to zero mean and unit variance before fitting,
    which improves numerical stability when columns span very different
    scales (e.g., RRP in dollars vs. binary holiday flag). Ridge penalty
    is applied in the standardized space and excludes the intercept.
    """

    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0, ddof=0)
        # Avoid division by zero for constant columns.
        self._std = np.where(self._std == 0, 1.0, self._std)
        Xs = (X - self._mean) / self._std
        # Add intercept column.
        Xa = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        if self.alpha > 0:
            n_params = Xa.shape[1]
            penalty = self.alpha * np.eye(n_params)
            penalty[0, 0] = 0.0   # don't penalize intercept
            beta = np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y)
        else:
            beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fit yet")
        X = np.asarray(X, dtype=float)
        Xs = (X - self._mean) / self._std
        return self.intercept_ + Xs @ self.coef_

    def __repr__(self) -> str:
        return f"NumpyOLS(alpha={self.alpha})"
