"""Figures for the thesis and the evaluation report (PNG, matplotlib without a display)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _pyplot().close(fig)
    return path


def training_curves(histories: dict[int, list[dict[str, Any]]], path: Path) -> Path:
    """Loss and validation AUC per epoch, one line per fold (section 11.4)."""
    plt = _pyplot()
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 4))
    for fold, history in sorted(histories.items()):
        epochs = [h["epoch"] + 1 for h in history]
        (line,) = ax_loss.plot(
            epochs, [h["train_loss"] for h in history], label=f"fold {fold} train"
        )
        ax_loss.plot(
            epochs,
            [h["val_loss"] for h in history],
            "--",
            color=line.get_color(),
            label=f"fold {fold} val",
        )
        ax_auc.plot(epochs, [h["val_auc"] for h in history], marker="o", ms=3, label=f"fold {fold}")
    ax_loss.set(xlabel="epoch", ylabel="loss (BCE)", title="Training and validation loss")
    ax_auc.set(xlabel="epoch", ylabel="AUC", title="Validation AUC", ylim=(0, 1))
    ax_auc.axhline(0.5, color="grey", lw=0.8, ls=":")
    for ax in (ax_loss, ax_auc):
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    return _save(fig, path)


def roc_figure(curves: Sequence[dict[str, Any]], path: Path, title: str = "ROC curve") -> Path:
    """curves: dicts with label, fpr, tpr, auc and optional auc_ci."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 5))
    for curve in curves:
        ci = curve.get("auc_ci")
        text = f"{curve['label']} (AUC {curve['auc']:.3f}" + (
            f", 95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else ")"
        )
        ax.plot(curve["fpr"], curve["tpr"], lw=2, label=text)
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1, label="chance")
    ax.set(
        xlabel="1 - specificity (false positive rate)", ylabel="sensitivity (true positive rate)"
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), title=title, aspect="equal")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def reliability_figure(
    bins: Sequence[dict[str, Any]], path: Path, ece: float, title: str = "Reliability diagram"
) -> Path:
    """bins: dicts with lower, upper, count, mean_predicted, observed (section 4.4)."""
    plt = _pyplot()
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]})
    used = [b for b in bins if b["count"]]
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1, label="perfect calibration")
    ax.plot(
        [b["mean_predicted"] for b in used],
        [b["observed"] for b in used],
        marker="o",
        lw=2,
        label=f"model (ECE {ece:.3f})",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="observed fraction TB", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    widths = [b["upper"] - b["lower"] for b in bins]
    hist.bar(
        [b["lower"] for b in bins],
        [b["count"] for b in bins],
        width=widths,
        align="edge",
        edgecolor="white",
    )
    hist.set(xlim=(0, 1), xlabel="predicted probability of TB", ylabel="cases")
    return _save(fig, path)


def confusion_figure(matrix: dict[str, int], path: Path, threshold: float) -> Path:
    plt = _pyplot()
    cells = np.array([[matrix["tp"], matrix["fn"]], [matrix["fp"], matrix["tn"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cells, cmap="Blues")
    for (i, j), value in np.ndenumerate(cells):
        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=14,
            color="white" if value > cells.max() / 2 else "black",
        )
    ax.set_xticks([0, 1], ["predicted TB", "predicted NTM"])
    ax.set_yticks([0, 1], ["actual TB", "actual NTM"])
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    return _save(fig, path)


def subgroup_figure(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    """Forest plot of subgroup AUCs with 95% CIs (section 4.4)."""
    plt = _pyplot()
    shown = [r for r in rows if r.get("auc") is not None]
    fig, ax = plt.subplots(figsize=(6, 0.4 * max(len(shown), 1) + 1))
    for y, row in enumerate(reversed(shown)):
        low, high = row.get("auc_ci") or (row["auc"], row["auc"])
        ax.plot([low, high], [y, y], color="tab:blue")
        ax.plot(row["auc"], y, "o", color="tab:blue")
    ax.set_yticks(
        range(len(shown)), [f"{r['group']}: {r['value']} (n={r['n']})" for r in reversed(shown)]
    )
    ax.axvline(0.5, color="grey", ls=":")
    ax.set(xlim=(0, 1), xlabel="AUC (95% CI)", title="Subgroup AUC")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path)


def score_histogram(labels: Sequence[int], probabilities: Sequence[float], path: Path) -> Path:
    plt = _pyplot()
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    edges = np.linspace(0, 1, 21)
    ax.hist(p[y], bins=edges, alpha=0.6, label=f"TB (n={y.sum()})")
    ax.hist(p[~y], bins=edges, alpha=0.6, label=f"NTM (n={(~y).sum()})")
    ax.axvline(0.5, color="black", ls="--", lw=1, label="threshold 0.50")
    ax.set(xlabel="predicted probability of TB", ylabel="cases", title="Predicted probabilities")
    ax.legend(fontsize=8)
    return _save(fig, path)
