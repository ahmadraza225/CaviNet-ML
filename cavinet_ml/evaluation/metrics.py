"""Binary classification metrics (TB = positive class). Phase 6 adds the full evaluation
(bootstrap intervals, calibration error, DeLong); these are shared with it."""

import numpy as np


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties counted half)."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    positives, negatives = s[y], s[~y]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positives, negatives]), kind="mergesort")
    values = np.concatenate([positives, negatives])[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:  # average ranks over ties
        j = i
        while j + 1 < values.size and values[j + 1] == values[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    rank_of = np.empty_like(ranks)
    rank_of[order] = ranks
    positive_ranks = rank_of[: positives.size].sum()
    u = positive_ranks - positives.size * (positives.size + 1) / 2
    return float(u / (positives.size * negatives.size))


def threshold_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(labels, dtype=bool)
    predicted = np.asarray(probabilities, dtype=np.float64) >= threshold
    tp = int(np.sum(predicted & y))
    tn = int(np.sum(~predicted & ~y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))

    def ratio(a: int, b: int) -> float:
        return float(a / b) if b else float("nan")

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "accuracy": ratio(tp + tn, y.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def brier_score(labels, probabilities) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean((p - y) ** 2))
