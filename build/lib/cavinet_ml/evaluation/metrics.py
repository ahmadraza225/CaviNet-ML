"""Binary classification metrics (TB = positive class), section 11.7."""

import numpy as np
from scipy.stats import rankdata


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties counted half)."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)  # average ranks over ties
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def threshold_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, float]:
    """Everything that depends on the decision threshold (predicted TB if p >= threshold)."""
    y = np.asarray(labels, dtype=bool)
    predicted = np.asarray(probabilities, dtype=np.float64) >= threshold
    tp = int(np.sum(predicted & y))
    tn = int(np.sum(~predicted & ~y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": _ratio(tn, tn + fn),
        "accuracy": _ratio(tp + tn, y.size),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def brier_score(labels, probabilities) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def log_loss(labels, probabilities, eps: float = 1e-12) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_curve(labels, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(false positive rate, true positive rate, thresholds), from (0, 0) to (1, 1)."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(s)[::-1]
    n_pos, n_neg = max(int(y.sum()), 1), max(int((~y).sum()), 1)
    tpr = [0.0] + [float(np.sum((s >= t) & y)) / n_pos for t in thresholds]
    fpr = [0.0] + [float(np.sum((s >= t) & ~y)) / n_neg for t in thresholds]
    return np.array(fpr), np.array(tpr), np.concatenate([[np.inf], thresholds])


def youden_threshold(labels, probabilities) -> float:
    """The threshold maximising sensitivity + specificity - 1 (section 11.5: reported for
    reference; the application always uses 0.50)."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    best = int(np.argmax(tpr[1:] - fpr[1:])) + 1
    return float(thresholds[best])


def calibration_bins(labels, probabilities, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-width bins of the predicted TB probability: mean prediction vs observed TB rate."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        inside = index == b
        count = int(inside.sum())
        bins.append(
            {
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": count,
                "mean_predicted": float(p[inside].mean()) if count else float("nan"),
                "observed": float(y[inside].mean()) if count else float("nan"),
            }
        )
    return bins


def expected_calibration_error(labels, probabilities, n_bins: int = 10) -> float:
    """Σ (bin share) × |observed TB rate − mean predicted probability| over equal-width bins."""
    n = len(np.asarray(labels))
    if n == 0:
        return float("nan")
    return float(
        sum(
            b["count"] / n * abs(b["observed"] - b["mean_predicted"])
            for b in calibration_bins(labels, probabilities, n_bins)
            if b["count"]
        )
    )
