"""Training building blocks shared by the demo model and (from Phase 6) the real training
toolkit: seeding, one training epoch, and prediction on preprocessed volumes."""

import random
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(indices: Sequence[int], size: int, rng: np.random.Generator | None = None):
    order = np.array(indices)
    if rng is not None:
        order = rng.permutation(order)
    for start in range(0, len(order), size):
        yield order[start : start + size]


def to_tensor(volumes: Iterable[np.ndarray]) -> torch.Tensor:
    """Preprocessed float16 volumes → float32 batch [N, 1, D, H, W]."""
    return torch.from_numpy(np.stack([np.asarray(v, dtype=np.float32) for v in volumes]))[:, None]


def train_epoch(
    model: nn.Module,
    volumes: Sequence[np.ndarray],
    labels: Sequence[int],
    indices: Sequence[int],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    rng: np.random.Generator,
    pos_weight: float | None = None,
) -> float:
    """One pass over `indices` with BCE-with-logits (TB = positive). Returns the mean loss."""
    model.train()
    weight = torch.tensor([pos_weight]) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    total, count = 0.0, 0
    for batch in batches(indices, batch_size, rng):
        x = to_tensor(volumes[i] for i in batch)
        y = torch.tensor([float(labels[i]) for i in batch])
        optimizer.zero_grad()
        loss = loss_fn(model(x).reshape(-1), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
        count += len(batch)
    return total / max(count, 1)


def recalibrate_batchnorm(
    model: nn.Module, volumes: Sequence[np.ndarray], indices: Sequence[int], batch_size: int = 4
) -> None:
    """Recompute BatchNorm running statistics as an exact average over `indices`. After a
    short training run the momentum-based statistics lag behind the weights; this makes
    eval-mode outputs match what the model learned."""
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return
    saved = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.train()
    with torch.no_grad():
        for batch in batches(indices, batch_size):
            model(to_tensor(volumes[i] for i in batch))
    for m, momentum in zip(norms, saved, strict=True):
        m.momentum = momentum
    model.eval()


def predict_logits(
    model: nn.Module, volumes: Sequence[np.ndarray], indices: Sequence[int], batch_size: int = 4
) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for batch in batches(indices, batch_size):
            out.append(model(to_tensor(volumes[i] for i in batch)).reshape(-1).numpy())
    return np.concatenate(out) if out else np.zeros(0)
