"""`cavinet-ml train --fold k` (FR-10.4): one cross-validation fold per section 11.4.

- Data: the fold's training patients (the other folds of the development set) and its
  validation patients, read from the preprocessed cache; the locked test set is never read.
- Loss: BCE with logits, TB positive, pos_weight = #NTM / #TB of the training fold.
- AdamW (lr 1e-4, weight decay 1e-4), 3 warm-up epochs then a cosine schedule.
- Batch 4 with mixed precision (AMP) and gradient accumulation to an effective 16. If the GPU
  runs out of memory the batch is halved (keeping the effective size) and the epoch restarts
  from the last checkpoint.
- Up to 60 epochs; early stopping on validation AUC with patience 12; the best checkpoint
  is kept. `last.pt` is written after every epoch, so running the same command again resumes.
- Logs: CSV and TensorBoard per fold, the configuration (YAML) and provenance (run.json),
  training curves (PNG), and the best model's out-of-fold logits (oof.csv) for calibration.
"""

import csv
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cavinet_ml.dataset.cache import cache_config, cache_file
from cavinet_ml.dataset.manifest import file_sha256, read_manifest
from cavinet_ml.dataset.source import DatasetError
from cavinet_ml.dataset.split import load_splits
from cavinet_ml.evaluation.metrics import roc_auc
from cavinet_ml.model.network import build_model
from cavinet_ml.provenance import environment, git_commit
from cavinet_ml.training.augment import air_value, augment_batch
from cavinet_ml.training.config import TrainingConfig
from cavinet_ml.training.loop import set_seed
from cavinet_ml.training.pretrained import initialise

LOG_COLUMNS = (
    "epoch",
    "lr",
    "train_loss",
    "val_loss",
    "val_auc",
    "best_val_auc",
    "batch_size",
    "seconds",
)


class TrainingError(Exception):
    """Training cannot start or continue; the message says what to do."""


class _ReduceBatch(Exception):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.size = size


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    best_val_auc: float
    epochs_run: int
    stopped_early: bool
    batch_size: int
    run_dir: Path


class CachedVolumes(torch.utils.data.Dataset):
    """Preprocessed float16 volumes [1, D, H, W] and labels from the cache."""

    def __init__(self, ids: list[str], labels: dict[str, int], cache_dir: Path) -> None:
        self.ids, self.labels, self.cache_dir = ids, labels, cache_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_id = self.ids[i]
        volume = np.load(cache_file(self.cache_dir, case_id))
        return torch.from_numpy(volume.astype(np.float16))[None], torch.tensor(
            float(self.labels[case_id])
        )


def lr_factor(epoch: int, warmup: int, max_epochs: int) -> float:
    """Linear warm-up over `warmup` epochs, then cosine decay to 0 at `max_epochs`."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    span = max(1, max_epochs - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(epoch - warmup, span) / span))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but PyTorch sees no GPU (see the runbook).")
    return torch.device(device)


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _save(obj: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def predict(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, amp: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(logits, labels) in loader order."""
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x.to(device, non_blocking=True).float())
            logits.append(out.float().reshape(-1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


@dataclass
class _Context:
    fold: int
    config: TrainingConfig
    device: torch.device
    amp: bool
    run_dir: Path
    train_ids: list[str]
    val_ids: list[str]
    labels: dict[str, int]
    cache_dir: Path
    fill_value: float
    pos_weight: float
    provenance: dict[str, Any]
    log: Callable[[str], None]


def _loader(ctx: _Context, ids: list[str], batch_size: int, shuffle: bool, seed: int):
    return torch.utils.data.DataLoader(
        CachedVolumes(ids, ctx.labels, ctx.cache_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=ctx.config.data.num_workers,
        pin_memory=ctx.device.type == "cuda",
    )


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError)


def _run(ctx: _Context, batch_size: int, restart_state: dict[str, Any] | None) -> FoldResult:
    config, device, log = ctx.config, ctx.device, ctx.log
    set_seed(config.seed)
    model = build_model(config.architecture())
    if restart_state is None:
        initialisation = initialise(model, config.model.pretrained, log=log)
    else:
        initialisation = restart_state["initialisation"]
        model.load_state_dict(restart_state["model"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_factor(e, config.schedule.warmup_epochs, config.epochs.max)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=ctx.amp)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ctx.pos_weight], device=device))
    val_loss_fn = nn.BCEWithLogitsLoss()

    state = {"epoch": 0, "best_val_auc": -1.0, "best_epoch": -1, "bad_epochs": 0, "history": []}
    if restart_state is not None:
        optimizer.load_state_dict(restart_state["optimizer"])
        scheduler.load_state_dict(restart_state["scheduler"])
        scaler.load_state_dict(restart_state["scaler"])
        state = {k: restart_state[k] for k in state}
        log(
            f"Fold {ctx.fold}: resuming at epoch {state['epoch'] + 1} with batch size {batch_size}."
        )
    accumulation = config.accumulation_steps(batch_size)

    writer = None
    if config.logging.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(ctx.run_dir / "tensorboard"))
    log_path = ctx.run_dir / "train_log.csv"
    val_loader = _loader(ctx, ctx.val_ids, batch_size, False, 0)
    stopped_early = False
    try:
        for epoch in range(state["epoch"], config.epochs.max):
            started = time.perf_counter()
            seed = config.seed * 1000 + ctx.fold * 100 + epoch
            augment_rng = _generator(seed, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, count = 0.0, 0
            loader = _loader(ctx, ctx.train_ids, batch_size, True, seed)
            try:
                for step, (x, y) in enumerate(loader):
                    x = augment_batch(
                        x.to(device, non_blocking=True),
                        config.augmentation,
                        augment_rng,
                        ctx.fill_value,
                    )
                    y = y.to(device).float()
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=ctx.amp
                    ):
                        logits = model(x).reshape(-1)
                    loss = loss_fn(logits.float(), y)
                    scaler.scale(loss / accumulation).backward()
                    if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    total += loss.item() * len(y)
                    count += len(y)
                val_logits, val_labels = predict(model, val_loader, device, ctx.amp)
            except Exception as error:
                if not _is_oom(error):
                    raise
                if not config.batch.auto_reduce_on_oom or batch_size == 1:
                    raise TrainingError(
                        f"The GPU ran out of memory with batch size {batch_size}. See the runbook "
                        "(close other GPU programs; a smaller input size needs re-approval)."
                    ) from error
                raise _ReduceBatch(max(1, batch_size // 2)) from error

            val_auc = roc_auc(val_labels, val_logits)
            val_loss = float(
                val_loss_fn(torch.from_numpy(val_logits), torch.from_numpy(val_labels)).item()
            )
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            first = state["best_epoch"] < 0  # always keep a checkpoint, even if AUC is undefined
            if first or (not math.isnan(val_auc) and val_auc > state["best_val_auc"] + 1e-9):
                best = val_auc if not math.isnan(val_auc) else -1.0
                state.update(best_val_auc=best, best_epoch=epoch, bad_epochs=0)
                _save(
                    {
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch,
                        "val_auc": val_auc,
                    },
                    ctx.run_dir / "best.pt",
                )
            else:
                state["bad_epochs"] += 1
            record = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": total / max(count, 1),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "best_val_auc": state["best_val_auc"],
                "batch_size": batch_size,
                "seconds": round(time.perf_counter() - started, 1),
            }
            state["history"].append(record)
            state["epoch"] = epoch + 1
            stopped_early = state["bad_epochs"] >= config.epochs.patience
            finished = stopped_early or state["epoch"] >= config.epochs.max
            _save(
                {
                    **state,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "batch_size": batch_size,
                    "initialisation": initialisation,
                    "finished": finished,
                    "stopped_early": stopped_early,
                    "config_sha256": config.sha256(),
                    "splits_sha256": ctx.provenance["splits_sha256"],
                },
                ctx.run_dir / "last.pt",
            )
            if config.logging.csv:
                new = not log_path.is_file()
                with log_path.open("a", newline="", encoding="utf-8") as handle:
                    out = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
                    if new:
                        out.writeheader()
                    out.writerow(record)
            if writer is not None:
                writer.add_scalar("loss/train", record["train_loss"], epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("auc/val", val_auc, epoch)
                writer.add_scalar("lr", lr, epoch)
                writer.flush()
            if config.logging.curves_png:
                from cavinet_ml.evaluation.figures import training_curves

                training_curves({ctx.fold: state["history"]}, ctx.run_dir / "curves.png")
            log(
                f"Fold {ctx.fold} epoch {epoch + 1}/{config.epochs.max}: train loss "
                f"{record['train_loss']:.4f}, val loss {val_loss:.4f}, val AUC {val_auc:.4f} "
                f"(best {state['best_val_auc']:.4f} at epoch {state['best_epoch'] + 1}), "
                f"{record['seconds']:.0f} s"
            )
            if stopped_early:
                log(
                    f"Fold {ctx.fold}: no improvement for {config.epochs.patience} epochs; "
                    "stopping early."
                )
                break
    finally:
        if writer is not None:
            writer.close()
    return _finish(ctx, initialisation, batch_size, stopped_early)


def _finish(ctx: _Context, initialisation: str, batch_size: int, stopped_early: bool) -> FoldResult:
    """Out-of-fold logits of the best checkpoint, and run.json."""
    last = _load(ctx.run_dir / "last.pt")
    best = _load(ctx.run_dir / "best.pt")
    model = build_model(ctx.config.architecture())
    model.load_state_dict(best["model"])
    model.to(ctx.device)
    loader = _loader(ctx, ctx.val_ids, batch_size, False, 0)
    logits, labels = predict(model, loader, ctx.device, ctx.amp)
    with (ctx.run_dir / "oof.csv").open("w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["case_id", "fold", "label", "logit"])
        for case_id, label, logit in zip(ctx.val_ids, labels, logits, strict=True):
            out.writerow([case_id, ctx.fold, int(label), f"{float(logit):.6f}"])
    result = FoldResult(
        fold=ctx.fold,
        best_epoch=int(best["epoch"]),
        best_val_auc=float(best["val_auc"]),
        epochs_run=int(last["epoch"]),
        stopped_early=bool(last.get("stopped_early", stopped_early)),
        batch_size=int(last["batch_size"]),
        run_dir=ctx.run_dir,
    )
    run = {
        **ctx.provenance,
        "fold": ctx.fold,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "initialisation": initialisation,
        "device": str(ctx.device),
        "amp": ctx.amp,
        "pos_weight": round(ctx.pos_weight, 6),
        "train_patients": {
            "n": len(ctx.train_ids),
            "tb": sum(ctx.labels[c] for c in ctx.train_ids),
        },
        "val_patients": {"n": len(ctx.val_ids), "tb": sum(ctx.labels[c] for c in ctx.val_ids)},
        "best_epoch": result.best_epoch + 1,
        "best_val_auc": result.best_val_auc,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "batch_size_used": result.batch_size,
    }
    (ctx.run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    ctx.log(
        f"Fold {ctx.fold} finished: best val AUC {result.best_val_auc:.4f} at epoch "
        f"{result.best_epoch + 1} of {result.epochs_run}. "
        f"Out-of-fold logits: {ctx.run_dir / 'oof.csv'}"
    )
    return result


def train_fold(
    fold: int,
    *,
    config: TrainingConfig,
    manifest_path: Path,
    splits_path: Path,
    cache_dir: Path,
    runs_dir: Path,
    device: str = "auto",
    restart: bool = False,
    log: Callable[[str], None] = print,
) -> FoldResult:
    splits = load_splits(splits_path)
    if not 0 <= fold < splits.n_folds:
        raise TrainingError(f"--fold must be between 0 and {splits.n_folds - 1}.")
    rows = {r["case_id"]: r for r in read_manifest(manifest_path)}
    manifest_sha = file_sha256(manifest_path)
    if splits.manifest_sha256 and splits.manifest_sha256 != manifest_sha:
        raise TrainingError(f"{splits_path} was made from a different manifest.csv.")
    train_ids, val_ids = splits.train_ids(fold), splits.val_ids(fold)
    missing = [c for c in train_ids + val_ids if not cache_file(cache_dir, c).is_file()]
    if missing:
        raise DatasetError(
            f"{len(missing)} cached volumes are missing ({', '.join(missing[:10])})."
        )
    torch_device = resolve_device(device)
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    run_dir = runs_dir / f"fold_{fold}"
    splits_sha = file_sha256(splits_path)
    if restart and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    last_path = run_dir / "last.pt"
    previous = _load(last_path) if last_path.is_file() else None
    if previous is not None:
        if previous["config_sha256"] != config.sha256() or previous["splits_sha256"] != splits_sha:
            raise TrainingError(
                f"{run_dir} holds a run with a different configuration or split; "
                "use --restart to train this fold again from the start."
            )
    (run_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")
    labels = {c: rows[c]["label"] for c in train_ids + val_ids}
    n_tb = sum(labels[c] == 1 for c in train_ids)
    if n_tb == 0 or n_tb == len(train_ids):
        raise TrainingError(f"Fold {fold}'s training patients are all one class.")
    pos_weight = (
        (len(train_ids) - n_tb) / n_tb
        if config.loss.pos_weight == "auto"
        else float(config.loss.pos_weight)
    )
    ctx = _Context(
        fold=fold,
        config=config,
        device=torch_device,
        amp=config.batch.amp and torch_device.type == "cuda",
        run_dir=run_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        labels=labels,
        cache_dir=cache_dir,
        fill_value=air_value(cache_config(cache_dir)),
        pos_weight=pos_weight,
        provenance={
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "config_sha256": config.sha256(),
            "manifest_sha256": manifest_sha,
            "splits_sha256": splits_sha,
            "environment": environment(),
        },
        log=log,
    )
    if previous is not None and previous.get("finished"):
        log(f"Fold {fold} already finished; use --restart to train it again.")
        return _finish(
            ctx,
            previous["initialisation"],
            int(previous["batch_size"]),
            bool(previous.get("stopped_early")),
        )
    batch_size = int(previous["batch_size"]) if previous else config.batch.size
    log(
        f"Fold {fold}: {len(train_ids)} training and {len(val_ids)} validation patients on "
        f"{torch_device} (AMP {'on' if ctx.amp else 'off'}), batch {batch_size} × "
        f"{config.accumulation_steps(batch_size)} accumulation steps."
    )
    while True:
        try:
            return _run(ctx, batch_size, previous)
        except _ReduceBatch as reduce:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            log(
                f"Fold {fold}: the GPU ran out of memory with batch size {batch_size}; "
                "retrying with "
                f"{reduce.size} (gradient accumulation keeps the effective batch at "
                f"{config.batch.effective_size})."
            )
            batch_size = reduce.size
            previous = _load(last_path) if last_path.is_file() else None
            if previous is not None:
                previous["batch_size"] = batch_size
