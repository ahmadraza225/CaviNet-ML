"""manifest.csv (FR-10.1): one row per patient with label, clinical values and scan details.

A sidecar manifest.json records where the data came from, whether it is the synthetic
look-alike, and every problem found while indexing.
"""

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cavinet_ml.dataset.patient_index import SYMPTOM_COLUMNS

MANIFEST_COLUMNS = (
    "case_id",
    "label",  # 1 = TB, 0 = NTM (from the folder, section 7.3)
    "label_name",
    "location",  # patient folder inside the dataset
    "age",
    "sex",  # male / female
    "age_band",  # <40, 40-59, >=60 (section 11.3)
    "index_status",  # ok, duplicate (listed twice with different values) or missing
    *SYMPTOM_COLUMNS,  # 1 present, 0 absent, blank unknown
    "symptoms_known",
    "files",
    "dicom_files",
    "series",
    "slices",  # slices in the series used (the one with the most slices, FR-04.3)
    "slice_thickness_mm",
    "slice_spacing_mm",
    "pixel_spacing_mm",
    "rows",
    "columns",
    "manufacturer",
    "manufacturer_group",
    "model",
    "kernel",
    "transfer_syntax",
    "compressed",
    "dicom_sex",
    "dicom_age",
    "notes",
)
AGE_BANDS = ("<40", "40-59", ">=60")
UNKNOWN = "unknown"

_MANUFACTURERS = (
    ("GE", "GE"),
    ("TOSHIBA", "TOSHIBA"),
    ("CANON", "CANON"),
    ("SIEMENS", "SIEMENS"),
    ("PHILIPS", "PHILIPS"),
    ("UNITED IMAGING", "UIH"),
    ("UIH", "UIH"),
    ("NEUSOFT", "NEUSOFT"),
    ("HITACHI", "HITACHI"),
)


def age_band(age: int | None) -> str:
    if age is None:
        return UNKNOWN
    if age < 40:
        return "<40"
    return "40-59" if age < 60 else ">=60"


def manufacturer_group(name: str | None) -> str:
    """ "GE MEDICAL SYSTEMS" -> "GE"; used for stratification and subgroup results."""
    text = " ".join((name or "").upper().replace(",", " ").split())
    if not text:
        return UNKNOWN
    for needle, group in _MANUFACTURERS:
        if text == needle or text.startswith(needle + " ") or f" {needle} " in f" {text} ":
            return group
    return text.split()[0]


def write_manifest(rows: Iterable[dict[str, Any]], path: Path) -> str:
    """Write manifest.csv and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in MANIFEST_COLUMNS})
    temporary.replace(path)
    return file_sha256(path)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _int(text: str) -> int | None:
    return int(float(text)) if text not in ("", None) else None


def _float(text: str) -> float | None:
    return float(text) if text not in ("", None) else None


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """manifest.csv rows with typed values (None for blanks)."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `cavinet-ml index` first.")
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["label"] = int(raw["label"])
            for key in ("age", "files", "dicom_files", "series", "slices", "rows", "columns"):
                row[key] = _int(raw[key])
            row["dicom_age"] = _int(raw["dicom_age"])
            for key in ("slice_thickness_mm", "slice_spacing_mm", "pixel_spacing_mm"):
                row[key] = _float(raw[key])
            for key in SYMPTOM_COLUMNS:
                row[key] = _int(raw[key])
            row["symptoms_known"] = raw["symptoms_known"] == "true"
            row["compressed"] = raw["compressed"] == "true"
            row["sex"] = raw["sex"] or None
            rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".json")


def read_summary(manifest_path: Path) -> dict[str, Any]:
    path = summary_path(manifest_path)
    return json.loads(path.read_text()) if path.is_file() else {}
