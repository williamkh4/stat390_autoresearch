"""
Locked validation metric for this project.

PRIMARY_METRIC = mean squared error of daily demand on the validation set.
This is the scalar the AutoResearch loop optimizes. Do not change it
mid-project: every experiment must be comparable on the same ruler.

Secondary metrics (RMSE, MAE) are reported for context only.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


PRIMARY_METRIC_NAME = "mse_demand"
PRIMARY_METRIC_DIRECTION = "minimize"


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def score_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return primary + secondary metrics in one dict, primary key first."""
    return {
        PRIMARY_METRIC_NAME: mse(y_true, y_pred),
        "rmse_demand": rmse(y_true, y_pred),
        "mae_demand": mae(y_true, y_pred),
    }
