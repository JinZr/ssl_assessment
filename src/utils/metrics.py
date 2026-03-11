from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Expected a 1D array")
    return array


def mse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    truth = _safe_array(y_true)
    pred = _safe_array(y_pred)
    return float(np.mean((truth - pred) ** 2))


def mae(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    truth = _safe_array(y_true)
    pred = _safe_array(y_pred)
    return float(np.mean(np.abs(truth - pred)))


def rmse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def lcc(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    truth = _safe_array(y_true)
    pred = _safe_array(y_pred)
    if truth.shape[0] < 2:
        return 0.0
    if np.std(truth) == 0 or np.std(pred) == 0:
        return 0.0
    return float(pearsonr(truth, pred).statistic)


def srcc(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    truth = _safe_array(y_true)
    pred = _safe_array(y_pred)
    if truth.shape[0] < 2:
        return 0.0
    if np.std(truth) == 0 or np.std(pred) == 0:
        return 0.0
    return float(spearmanr(truth, pred).statistic)


def ccc(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    truth = _safe_array(y_true)
    pred = _safe_array(y_pred)
    covariance = np.cov(truth, pred, bias=True)[0, 1]
    denom = np.var(truth) + np.var(pred) + (np.mean(truth) - np.mean(pred)) ** 2
    if denom == 0:
        return 0.0
    return float((2 * covariance) / denom)


@dataclass
class MetricBundle:
    mse: float
    lcc: float
    srcc: float
    mae: float
    ccc: float
    rmse: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "lcc": self.lcc,
            "srcc": self.srcc,
            "mae": self.mae,
            "ccc": self.ccc,
            "rmse": self.rmse,
        }


def compute_metrics(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> MetricBundle:
    return MetricBundle(
        mse=mse(y_true, y_pred),
        lcc=lcc(y_true, y_pred),
        srcc=srcc(y_true, y_pred),
        mae=mae(y_true, y_pred),
        ccc=ccc(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
    )


def bootstrap_ci(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float]:
    truth = _safe_array(y_true)
    pred = _safe_array(y_pred)
    rng = np.random.default_rng(seed)
    metrics = np.zeros(n_bootstrap, dtype=np.float64)
    n = truth.shape[0]
    for index in range(n_bootstrap):
        sample_ids = rng.integers(0, n, size=n)
        metrics[index] = metric_fn(truth[sample_ids], pred[sample_ids])
    lower = np.quantile(metrics, alpha / 2)
    upper = np.quantile(metrics, 1 - alpha / 2)
    return float(lower), float(upper)


def paired_permutation_test(
    y_true: np.ndarray | list[float],
    pred_a: np.ndarray | list[float],
    pred_b: np.ndarray | list[float],
    n_permutations: int = 10_000,
    seed: int = 13,
) -> float:
    truth = _safe_array(y_true)
    first = _safe_array(pred_a)
    second = _safe_array(pred_b)
    observed = np.mean((truth - first) ** 2 - (truth - second) ** 2)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        swap_mask = rng.random(truth.shape[0]) < 0.5
        left = np.where(swap_mask, second, first)
        right = np.where(swap_mask, first, second)
        diff = np.mean((truth - left) ** 2 - (truth - right) ** 2)
        if abs(diff) >= abs(observed):
            count += 1
    return float((count + 1) / (n_permutations + 1))
