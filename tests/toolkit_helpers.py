"""A small, fast training-toolkit workspace for tests: manifest, a cache of 32³ volumes with a
learnable TB/NTM signal, a QC report and a 2-fold split. No patient data."""

import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from cavinet_ml.config import PreprocessingConfig
from cavinet_ml.dataset.cache import QC_COLUMNS, cache_file, check_cache_config
from cavinet_ml.dataset.manifest import age_band, summary_path, write_manifest
from cavinet_ml.dataset.patient_index import SYMPTOM_COLUMNS
from cavinet_ml.dataset.split import split_dataset
from cavinet_ml.workspace import Workspace

SIZE = 32
CONFIG = PreprocessingConfig(output_size=(SIZE, SIZE, SIZE))


@contextmanager
def narrow_network():
    """The section 11.2 ResNet-18 layout with narrow layers, so unit tests of the training
    mechanics stay fast and write small checkpoints. The end-to-end test uses the full size."""
    from cavinet_ml.model.network import DEFAULT_ARCHITECTURE
    from cavinet_ml.training import config as training_config

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            training_config._ARCHITECTURES,
            "monai_resnet18",
            {**DEFAULT_ARCHITECTURE, "block_inplanes": [8, 16, 32, 64]},
        )
        yield


SCANNERS = (("GE MEDICAL SYSTEMS", "GE", "LUNG", 1.25), ("TOSHIBA", "TOSHIBA", "FC52", 1.0))


def manifest_rows(n_tb: int, n_ntm: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tb + n_ntm):
        tb = i < n_tb
        case_id = f"TB_{i + 1:03d}" if tb else f"Case_{i - n_tb + 1:03d}"
        age = int(rng.normal(45 if tb else 62, 8))
        manufacturer, group, kernel, thickness = SCANNERS[i % 2]
        rows.append(
            {
                "case_id": case_id,
                "label": int(tb),
                "label_name": "TB" if tb else "NTM",
                "location": f"{'TB' if tb else 'NTM'}/{case_id}",
                "age": age,
                "sex": "male" if rng.random() < 0.6 else "female",
                "age_band": age_band(age),
                "index_status": "ok",
                **{c: int(rng.random() < 0.3) for c in SYMPTOM_COLUMNS},
                "symptoms_known": True,
                "files": 60,
                "dicom_files": 60,
                "series": 1,
                "slices": 60,
                "slice_thickness_mm": thickness,
                "slice_spacing_mm": thickness,
                "pixel_spacing_mm": 0.7,
                "rows": 512,
                "columns": 512,
                "manufacturer": manufacturer,
                "manufacturer_group": group,
                "model": "Test",
                "kernel": kernel,
                "transfer_syntax": "Explicit VR Little Endian",
                "compressed": False,
                "dicom_sex": None,
                "dicom_age": None,
                "notes": "",
            }
        )
    return rows


def volume(label: int, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    v = rng.normal(0.3, 0.05, (size, size, size)).astype(np.float32)
    if label:  # a bright blob in the upper half for TB
        v[size // 2 :, size // 4 : size // 2, size // 4 : size // 2] += 0.5
    return np.clip(v, 0, 1).astype(np.float16)


def make_workspace(
    root: Path,
    *,
    n_tb: int = 12,
    n_ntm: int = 8,
    folds: int = 2,
    synthetic: bool = True,
    size: int = SIZE,
) -> Workspace:
    """Volumes of `size`³; at least 48 when training with batch size 1 (BatchNorm needs more
    than one value per channel in the last block; real 128³ inputs give 4³)."""
    ws = Workspace(work=root / "work", splits=root / "splits.json", docs=root / "docs")
    rows = manifest_rows(n_tb, n_ntm)
    write_manifest(rows, ws.manifest)
    summary_path(ws.manifest).write_text(
        json.dumps({"dataset": {"path": str(root), "kind": "folder", "synthetic": synthetic}}),
        encoding="utf-8",
    )
    check_cache_config(ws.cache, PreprocessingConfig(output_size=(size, size, size)))
    rng = np.random.default_rng(1)
    for row in rows:
        np.save(cache_file(ws.cache, row["case_id"]), volume(row["label"], rng, size))
    ws.qc_report.parent.mkdir(parents=True, exist_ok=True)
    with ws.qc_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"case_id": row["case_id"], "status": "ok", "used_fallback": "false"})
    split_dataset(ws.manifest, ws.qc_report, ws.splits, n_folds=folds, log=lambda m: None)
    return ws
