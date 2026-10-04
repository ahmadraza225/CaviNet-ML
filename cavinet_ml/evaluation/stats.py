"""Confidence intervals and the DeLong test (sections 4.3 and 11.7).

- Bootstrap: 2,000 resamples, stratified (TB and NTM cases resampled separately, so every
  resample has both classes), percentile 95% intervals; every metric uses the same resamples.
- DeLong (1988), with the fast midrank algorithm of Sun & Xu (2014): variance of an AUC and
  the test of two correlated AUCs measured on the same patients (H2).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import norm, rankdata

from cavinet_ml.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    roc_auc,
    threshold_metrics,
)

N_BOOTSTRAP = 2000
SEED = 42

Statistic = Callable[[np.ndarray, np.ndarray], float]


def bootstrap(
    labels,
    scores,
    statistics: dict[str, Statistic],
    *,
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """{name: {"value": statistic on all cases, "ci": [low, high]}}."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in statistics}
    for _ in range(n):
        index = np.concatenate(
            [
                rng.choice(positives, positives.size, replace=True),
                rng.choice(negatives, negatives.size, replace=True),
            ]
        )
        for name, fn in statistics.items():
            samples[name].append(fn(y[index], s[index]))
    result = {}
    for name, fn in statistics.items():
        values = np.asarray(samples[name], dtype=np.float64)
        values = values[~np.isnan(values)]
        low, high = (
            np.quantile(values, [alpha / 2, 1 - alpha / 2]) if values.size else (np.nan, np.nan)
        )
        result[name] = {"value": float(fn(y, s)), "ci": [float(low), float(high)]}
    return result


THRESHOLD_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "accuracy",
    "balanced_accuracy",
    "f1",
)


def metrics_with_ci(
    labels, probabilities, threshold: float = 0.5, *, n: int = N_BOOTSTRAP, seed: int = SEED
) -> dict[str, dict[str, Any]]:
    """Every section 11.7 metric of predicted probabilities, each with its 95% CI."""

    def at(name: str) -> Statistic:
        return lambda y, p: threshold_metrics(y, p, threshold)[name]

    statistics: dict[str, Statistic] = {
        "auc": roc_auc,
        **{name: at(name) for name in THRESHOLD_METRICS},
        "brier": brier_score,
        "ece": expected_calibration_error,
    }
    return bootstrap(labels, probabilities, statistics, n=n, seed=seed)


def _fast_delong(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """scores: [k, m + n], positives first. Returns (AUCs [k], covariance [k, k])."""
    m = n_pos
    n = scores.shape[1] - m
    positives, negatives = scores[:, :m], scores[:, m:]
    tx = np.vstack([rankdata(row) for row in positives])
    ty = np.vstack([rankdata(row) for row in negatives])
    tz = np.vstack([rankdata(row) for row in scores])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _ordered(labels, *score_sets) -> tuple[np.ndarray, int]:
    y = np.asarray(labels, dtype=bool)
    if y.all() or not y.any():
        raise ValueError("both classes are needed")
    order = np.concatenate([np.flatnonzero(y), np.flatnonzero(~y)])
    return np.vstack([np.asarray(s, dtype=np.float64)[order] for s in score_sets]), int(y.sum())


def delong_ci(labels, scores, alpha: float = 0.05) -> dict[str, Any]:
    stacked, m = _ordered(labels, scores)
    aucs, cov = _fast_delong(stacked, m)
    se = float(np.sqrt(cov[0, 0]))
    z = norm.ppf(1 - alpha / 2)
    return {
        "auc": float(aucs[0]),
        "se": se,
        "ci": [float(aucs[0] - z * se), float(aucs[0] + z * se)],
    }


def delong_test(labels, scores_a, scores_b, alpha: float = 0.05) -> dict[str, Any]:
    """Two-sided DeLong test of AUC(a) = AUC(b) on the same patients."""
    stacked, m = _ordered(labels, scores_a, scores_b)
    aucs, cov = _fast_delong(stacked, m)
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    difference = float(aucs[0] - aucs[1])
    se = float(np.sqrt(max(variance, 0.0)))
    if se == 0:
        z, p = (0.0, 1.0) if difference == 0 else (float("inf"), 0.0)
    else:
        z = difference / se
        p = float(2 * norm.sf(abs(z)))
    q = norm.ppf(1 - alpha / 2)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "difference": difference,
        "se": se,
        "z": float(z),
        "p_value": p,
        "ci": [difference - q * se, difference + q * se],
        "n": int(stacked.shape[1]),
        "n_tb": m,
    }
