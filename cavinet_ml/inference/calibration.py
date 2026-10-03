"""Section 11.5: temperature scaling. Training fits the temperature on out-of-fold logits;
the application divides the ensemble logit by it."""

import math

import numpy as np


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def calibrated_probability(logit: float, temperature: float) -> float:
    return sigmoid(logit / temperature)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    # log(1 + e^z) - y z, computed stably
    return float(np.mean(np.logaddexp(0.0, z) - labels * z))


def fit_temperature(logits, labels, low: float = 0.05, high: float = 20.0) -> float:
    """Temperature minimising the negative log-likelihood (golden-section search on log T)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if z.size == 0 or np.all(y == y[0]):
        return 1.0
    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(80):
        if _nll(z, y, math.exp(c)) < _nll(z, y, math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - ratio * (b - a), a + ratio * (b - a)
    return round(math.exp((a + b) / 2), 6)
