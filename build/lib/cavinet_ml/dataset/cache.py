"""`cavinet-ml preprocess` (FR-10.2): section 11.1 on every patient, into a float16 cache.

- One patient at a time: only that patient's files are copied out of the zip into a
  temporary folder, which is deleted afterwards. The zip is never extracted as a whole.
- The same code as the application: `cavinet_ml.io.load_series` (step 1) and
  `cavinet_ml.preprocessing.preprocess` (steps 2-7) with the lungmask R231 segmenter.
- Parallel worker processes; lungmask runs on the GPU when there is one.
- Resumable: every finished patient is appended to the QC report at once; a new run skips
  patients already done (and failed ones unless --retry-failed).
- QC report (CSV): status, failure reason, warnings (lung segmentation fallback), lung
  volume, crop size and timings per patient; a small QC image per patient shows the crop.
"""

import csv
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cavinet_ml.config import DEFAULT_PREPROCESSING, PreprocessingConfig
from cavinet_ml.dataset.manifest import read_manifest, read_summary
from cavinet_ml.dataset.parallel import run_parallel
from cavinet_ml.dataset.source import CaseEntry, DatasetError, DatasetSource, open_source

QC_COLUMNS = (
    "case_id",
    "status",  # ok or failed
    "error",
    "warnings",
    "lung_volume_ml",
    "used_fallback",
    "slices_loaded",
    "original_size",  # x × y × z voxels
    "original_spacing_mm",
    "crop_size_mm",  # z × y × x extent of the crop
    "input_mean",  # mean of the 128³ model input (0-1); near 0 or 1 means a bad crop
    "seconds",
    "finished_at",
)
CONFIG_FILE = "preprocessing.json"
SUCCESS_TARGET = 0.98  # section 15.9: at least 98% of scans preprocessed


@dataclass
class PreprocessSummary:
    total: int
    ok: int
    failed: dict[str, str]
    fallback: list[str]
    skipped: int
    pending: int  # in the manifest but not processed yet (e.g. after --limit)
    seconds: float

    @property
    def success_rate(self) -> float:
        return self.ok / self.total if self.total else 0.0


def cache_file(cache_dir: Path, case_id: str) -> Path:
    return cache_dir / f"{case_id}.npy"


def load_volume(cache_dir: Path, case_id: str, mmap: bool = True) -> np.ndarray:
    return np.load(cache_file(cache_dir, case_id), mmap_mode="r" if mmap else None)


def read_qc(path: Path) -> dict[str, dict[str, str]]:
    """Latest QC row per patient."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _write_qc(rows: dict[str, dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS)
        writer.writeheader()
        for case_id in sorted(rows, key=_order):
            writer.writerow({k: rows[case_id].get(k, "") for k in QC_COLUMNS})
    temporary.replace(path)


def _order(case_id: str) -> tuple[int, int]:
    prefix, number = case_id.split("_")
    return (0 if prefix == "TB" else 1, int(number))


def check_cache_config(cache_dir: Path, config: PreprocessingConfig) -> None:
    """The cache must hold volumes made with one set of section 11.1 parameters."""
    path = cache_dir / CONFIG_FILE
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != config.to_dict():
            raise DatasetError(
                f"{cache_dir} holds volumes made with different preprocessing parameters "
                f"({path}). Use a new cache folder."
            )
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")


def cache_config(cache_dir: Path) -> PreprocessingConfig:
    path = cache_dir / CONFIG_FILE
    if not path.is_file():
        raise DatasetError(f"{path} not found; run `cavinet-ml preprocess` first.")
    return PreprocessingConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def qc_image(volume: np.ndarray, path: Path) -> None:
    """Middle axial and coronal slices and a sagittal slice through one lung of the model
    input, side by side (head up, anterior up)."""
    from PIL import Image

    v = np.asarray(volume, dtype=np.float32)
    d, h, w = v.shape
    axial = v[d // 2]
    coronal = v[::-1, h // 2, :]
    sagittal = v[::-1, :, w // 4]
    strip = np.concatenate([axial, coronal, sagittal], axis=1)
    Image.fromarray((np.clip(strip, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


# --- worker processes ---------------------------------------------------------------------

_state: dict[str, Any] = {}


def _init_worker(
    data: str,
    cache_dir: str,
    images_dir: str,
    tmp_dir: str,
    weights: str | None,
    force_cpu: bool,
    threads: int,
    config: dict[str, Any],
) -> None:
    import torch

    from cavinet_ml import hide_library_notices
    from cavinet_ml.preprocessing.lungmask_segmenter import (
        BATCH_SIZE,
        GPU_BATCH_SIZE,
        LungmaskSegmenter,
    )

    hide_library_notices()
    logging.getLogger("lungmask").setLevel(logging.WARNING)  # no per-scan chatter
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalWarningDisplay(False)  # failures are reported in the QC report
    torch.set_num_threads(max(1, threads))
    segmenter = None
    if weights:
        batch = BATCH_SIZE if force_cpu else GPU_BATCH_SIZE
        segmenter = LungmaskSegmenter(weights, batch, force_cpu=force_cpu)
    _state.update(
        source=open_source(data),
        cache_dir=Path(cache_dir),
        images_dir=Path(images_dir),
        tmp_dir=Path(tmp_dir),
        segmenter=segmenter,
        config=PreprocessingConfig.from_dict(config),
    )


def _process(case: CaseEntry) -> dict[str, Any]:
    return preprocess_case(
        case,
        _state["source"],
        _state["segmenter"],
        _state["config"],
        cache_dir=_state["cache_dir"],
        images_dir=_state["images_dir"],
        tmp_dir=_state["tmp_dir"],
    )


def preprocess_case(
    case: CaseEntry,
    source: DatasetSource,
    segmenter: Any,
    config: PreprocessingConfig,
    *,
    cache_dir: Path,
    images_dir: Path,
    tmp_dir: Path,
) -> dict[str, Any]:
    """Section 11.1 for one patient; returns its QC row (never raises for a bad scan)."""
    from cavinet_ml.io.dicom import DicomLoadError, load_series
    from cavinet_ml.preprocessing.pipeline import preprocess

    started = time.perf_counter()
    row: dict[str, Any] = {"case_id": case.case_id}
    workdir = Path(tempfile.mkdtemp(prefix=f"{case.case_id}-", dir=tmp_dir))
    try:
        source.extract_case(case, workdir)
        image = load_series(workdir, strict=False)
        prepared = preprocess(image, segmenter, config)
        target = cache_file(cache_dir, case.case_id)
        temporary = target.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, prepared.volume)
        temporary.replace(target)
        qc_image(prepared.volume, images_dir / f"{case.case_id}.png")
        spacing = config.target_spacing_mm
        crop = "×".join(f"{(stop - start) * spacing:.0f}" for start, stop in prepared.crop)
        row.update(
            status="ok",
            warnings="; ".join(prepared.warnings),
            lung_volume_ml=prepared.lung.volume_ml,
            used_fallback="true" if prepared.lung.used_fallback else "false",
            slices_loaded=image.GetDepth(),
            original_size="×".join(str(n) for n in image.GetSize()),
            original_spacing_mm="×".join(f"{s:.3g}" for s in image.GetSpacing()),
            crop_size_mm=crop,
            input_mean=f"{float(np.mean(prepared.volume, dtype=np.float32)):.3f}",
        )
    except DicomLoadError as error:
        row.update(status="failed", error=str(error))
    except Exception as error:  # noqa: BLE001 - one bad scan never stops the run
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    row["seconds"] = f"{time.perf_counter() - started:.1f}"
    row["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return row


# --- the command ----------------------------------------------------------------------------


def resolve_lungmask_weights(
    weights: str | None, *, use_lungmask: bool, download: bool, log: Callable[[str], None]
) -> str | None:
    from cavinet_ml.preprocessing.lungmask_segmenter import DEFAULT_WEIGHTS_PATH, download_r231

    if not use_lungmask:
        log(
            "WARNING: --no-lungmask: every scan will be cropped to the body outline (fallback). "
            "Use this only for tests, never for the real dataset."
        )
        return None
    path = Path(weights or os.environ.get("LUNGMASK_WEIGHTS") or DEFAULT_WEIGHTS_PATH).expanduser()
    if not path.is_file():
        if not download:
            raise DatasetError(f"lungmask weights not found at {path}.")
        log(f"Downloading the lungmask R231 weights to {path} …")
        download_r231(path)
    return str(path)


def preprocess_dataset(
    manifest_path: Path,
    *,
    cache_dir: Path,
    qc_path: Path,
    data: Path | str | None = None,
    workers: int = 1,
    lungmask_weights: str | None = None,
    use_lungmask: bool = True,
    download_weights: bool = True,
    device: str = "auto",
    only: list[str] | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING,
    log: Callable[[str], None] = print,
) -> PreprocessSummary:
    started = time.perf_counter()
    rows = read_manifest(manifest_path)
    data = Path(data or read_summary(manifest_path).get("dataset", {}).get("path", ""))
    if not str(data) or not data.exists():
        raise DatasetError("The dataset location is unknown; pass it with --data.")
    check_cache_config(cache_dir, config)
    images_dir = qc_path.parent / "qc_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cache_dir.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wanted = [r["case_id"] for r in rows]
    if only:
        unknown = sorted(set(only) - set(wanted))
        if unknown:
            raise DatasetError(f"Not in the manifest: {', '.join(unknown)}")
        wanted = [c for c in wanted if c in set(only)]
    qc = read_qc(qc_path)
    done = {
        c
        for c in wanted
        if c in qc
        and (
            (qc[c]["status"] == "ok" and cache_file(cache_dir, c).is_file())
            or (qc[c]["status"] == "failed" and not retry_failed)
        )
    }
    todo = [c for c in wanted if c not in done]
    if limit is not None:
        todo = todo[:limit]

    with open_source(data) as source:
        entries = {c.case_id: c for c in source.layout().cases}
    missing = [c for c in todo if c not in entries]
    if missing:
        raise DatasetError(f"Patient folders not found in {data}: {', '.join(missing[:10])}")

    import torch

    use_gpu = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    weights = resolve_lungmask_weights(
        lungmask_weights, use_lungmask=use_lungmask, download=download_weights, log=log
    )
    workers = max(1, workers)
    threads = max(1, (os.cpu_count() or 1) // workers)
    log(
        f"{len(wanted)} patients in the manifest: {len(done)} already done, {len(todo)} to "
        f"preprocess with {workers} worker(s); lung segmentation on the "
        f"{'GPU' if use_gpu and weights else 'CPU'}."
    )

    qc_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not qc_path.is_file()
    with qc_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS)
        if new_file:
            writer.writeheader()
        results = run_parallel(
            _process,
            [entries[c] for c in todo],
            workers=workers,
            initializer=_init_worker,
            initargs=(
                str(data),
                str(cache_dir),
                str(images_dir),
                str(tmp_dir),
                weights,
                not use_gpu,
                threads,
                config.to_dict(),
            ),
        )
        for number, row in enumerate(results, start=1):
            writer.writerow({k: row.get(k, "") for k in QC_COLUMNS})
            handle.flush()
            qc[row["case_id"]] = row
            elapsed = time.perf_counter() - started
            remaining = elapsed / number * (len(todo) - number)
            detail = (
                f"lungs {row['lung_volume_ml']} mL"
                + (" FALLBACK" if row["used_fallback"] == "true" else "")
                if row["status"] == "ok"
                else row["error"]
            )
            log(
                f"[{number}/{len(todo)}] {row['case_id']} {row['status']} {row['seconds']} s, "
                f"{detail} (about {remaining / 60:.0f} min left)"
            )

    rows_now = {c: qc[c] for c in wanted if c in qc}
    _write_qc({**read_qc(qc_path), **rows_now}, qc_path)
    ok = [c for c, r in rows_now.items() if r["status"] == "ok"]
    failed = {c: r["error"] for c, r in rows_now.items() if r["status"] == "failed"}
    fallback = [c for c in ok if rows_now[c]["used_fallback"] == "true"]
    summary = PreprocessSummary(
        total=len(wanted),
        ok=len(ok),
        failed=failed,
        fallback=fallback,
        skipped=len(done),
        pending=len(wanted) - len(ok) - len(failed),
        seconds=round(time.perf_counter() - started, 1),
    )
    log(
        f"Preprocessed {summary.ok} of {summary.total} patients ({summary.success_rate:.1%}; "
        f"target ≥ {SUCCESS_TARGET:.0%}) in {summary.seconds / 60:.1f} min. QC report: {qc_path}"
    )
    if summary.pending:
        log(f"  {summary.pending} patients are not processed yet; run the command again.")
    for case_id, error in failed.items():
        log(f"  FAILED {case_id}: {error}")
    if fallback:
        log(f"  Lung segmentation fallback used for {len(fallback)}: {', '.join(fallback[:20])}")
    return summary
