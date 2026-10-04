"""`cavinet-ml index` (FR-10.1): dataset + PatientIndex.xlsx -> manifest.csv.

For every patient folder the DICOM headers are read (pixel data is skipped, nothing is
extracted to disk) to find the series with the most slices and its scan details. The label
comes from the folder (TB = 1, NTM = 0) and is cross-checked with the sheet the patient is
listed in. Patients whose index row is missing or ambiguous keep their scan; their age and
sex are taken from the DICOM header when present and their symptoms are left unknown.
"""

import json
import os
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cavinet_ml import __version__
from cavinet_ml.dataset.manifest import age_band, manufacturer_group, summary_path, write_manifest
from cavinet_ml.dataset.parallel import run_parallel
from cavinet_ml.dataset.patient_index import (
    SYMPTOM_COLUMNS,
    PatientIndex,
    read_patient_index,
)
from cavinet_ml.dataset.source import (
    INDEX_FILE_NAME,
    CaseEntry,
    DatasetError,
    DatasetSource,
    open_source,
)

MIN_SLICES = 50  # FR-04.2, the application's rule; noted, not enforced, in training
MAX_SLICE_SPACING_MM = 5.0
_TAGS = [
    "SeriesInstanceUID",
    "Modality",
    "SliceThickness",
    "PixelSpacing",
    "Rows",
    "Columns",
    "Manufacturer",
    "ManufacturerModelName",
    "ConvolutionKernel",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PatientSex",
    "PatientAge",
]


def _most_common(values: list[Any]) -> Any:
    present = [v for v in values if v not in (None, "")]
    return Counter(present).most_common(1)[0][0] if present else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list | tuple) or type(value).__name__ == "MultiValue":
        return "\\".join(str(v).strip() for v in value) or None
    return str(value).strip() or None


def _dicom_age(value: Any) -> int | None:
    text = _text(value) or ""
    if len(text) == 4 and text[:3].isdigit() and text[3] in "YMWD":
        number = int(text[:3])
        return number if text[3] == "Y" else 0
    return None


def _dicom_sex(value: Any) -> str | None:
    return {"M": "male", "F": "female"}.get((_text(value) or "").upper())


def _normal(orientation: Any) -> tuple[float, float, float] | None:
    try:
        r = [float(v) for v in orientation]
    except (TypeError, ValueError):
        return None
    if len(r) != 6:
        return None
    a, b = r[:3], r[3:]
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def scan_details(source: DatasetSource, case: CaseEntry) -> dict[str, Any]:
    """Header-only summary of one patient's folder."""
    import pydicom  # training extra

    series: dict[str, list[Any]] = {}
    unreadable = 0
    for name in case.files:
        try:
            with source.open(name) as handle:
                ds = pydicom.dcmread(handle, stop_before_pixels=True, specific_tags=_TAGS)
        except Exception:  # noqa: BLE001 - any unreadable file is just counted
            unreadable += 1
            continue
        uid = _text(ds.get("SeriesInstanceUID"))
        if not uid:
            unreadable += 1
            continue
        series.setdefault(uid, []).append(ds)

    details: dict[str, Any] = {"files": len(case.files), "notes": []}
    notes: list[str] = details["notes"]
    if unreadable:
        notes.append(f"{unreadable} of {len(case.files)} files are not readable DICOM images")
    if not series:
        notes.append("no DICOM images found")
        details.update(dicom_files=0, series=0, slices=0)
        return details
    chosen = max(sorted(series), key=lambda uid: len(series[uid]))
    images = series[chosen]
    details["dicom_files"] = sum(len(v) for v in series.values())
    details["series"] = len(series)
    details["slices"] = len(images)
    if len(series) > 1:
        notes.append(f"{len(series)} series; the one with the most slices ({len(images)}) is used")

    modality = _most_common([_text(ds.get("Modality")) for ds in images])
    if modality != "CT":
        notes.append(f"modality is {modality or 'missing'}, not CT")
    thickness = _most_common([_text(ds.get("SliceThickness")) for ds in images])
    details["slice_thickness_mm"] = float(thickness) if thickness else None
    normal = _normal(
        _most_common([tuple(ds.get("ImageOrientationPatient") or ()) for ds in images])
    )
    if normal is not None:
        if abs(normal[2]) < 0.9:
            notes.append("slices are not axial")
        positions = sorted(
            {
                round(
                    sum(float(p) * n for p, n in zip(ds.ImagePositionPatient, normal, strict=True)),
                    3,
                )
                for ds in images
                if ds.get("ImagePositionPatient") is not None
            }
        )
        gaps = sorted(b - a for a, b in zip(positions, positions[1:], strict=False))
        details["slice_spacing_mm"] = round(gaps[len(gaps) // 2], 3) if gaps else None
    spacing = details.get("slice_spacing_mm")
    if spacing and spacing > MAX_SLICE_SPACING_MM:
        notes.append(f"slice spacing {spacing:g} mm is more than {MAX_SLICE_SPACING_MM:g} mm")
    if len(images) < MIN_SLICES:
        notes.append(f"only {len(images)} slices (the application needs at least {MIN_SLICES})")
    pixel = _most_common([tuple(ds.get("PixelSpacing") or ()) for ds in images])
    details["pixel_spacing_mm"] = float(pixel[0]) if pixel else None
    details["rows"] = _most_common([ds.get("Rows") for ds in images])
    details["columns"] = _most_common([ds.get("Columns") for ds in images])
    details["manufacturer"] = _most_common([_text(ds.get("Manufacturer")) for ds in images])
    details["model"] = _most_common([_text(ds.get("ManufacturerModelName")) for ds in images])
    details["kernel"] = _most_common([_text(ds.get("ConvolutionKernel")) for ds in images])
    syntax = _most_common([str(ds.file_meta.get("TransferSyntaxUID", "")) for ds in images])
    if syntax:
        uid = pydicom.uid.UID(syntax)
        details["transfer_syntax"] = uid.name
        details["compressed"] = bool(uid.is_compressed)
    details["dicom_sex"] = _dicom_sex(_most_common([_text(ds.get("PatientSex")) for ds in images]))
    details["dicom_age"] = _dicom_age(_most_common([_text(ds.get("PatientAge")) for ds in images]))
    return details


# Worker processes open the dataset once and keep it.
_worker_source: DatasetSource | None = None


def _init_worker(path: str) -> None:
    global _worker_source
    _worker_source = open_source(path)


def _worker_details(case: CaseEntry) -> tuple[str, dict[str, Any]]:
    assert _worker_source is not None
    return case.case_id, scan_details(_worker_source, case)


def manifest_row(case: CaseEntry, details: dict[str, Any], index: PatientIndex) -> dict[str, Any]:
    notes = list(details.get("notes", []))
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "label": case.label,
        "label_name": case.label_name,
        "location": case.location,
        **{k: v for k, v in details.items() if k != "notes"},
    }
    record = index.records.get(case.case_id)
    if record is not None:
        row.update(
            index_status="ok",
            clinical_source="index",
            sex=record.sex,
            age=record.age,
            symptoms_known=True,
            **record.symptoms,
        )
        dicom_age = details.get("dicom_age")
        if record.age is not None and dicom_age is not None and abs(record.age - dicom_age) > 1:
            notes.append(
                f"age {record.age} in PatientIndex.xlsx but {dicom_age} in the DICOM header"
            )
        dicom_sex = details.get("dicom_sex")
        if record.sex and dicom_sex and record.sex != dicom_sex:
            notes.append(
                f"sex {record.sex} in PatientIndex.xlsx but {dicom_sex} in the DICOM header"
            )
    else:
        status = "duplicate" if case.case_id in index.duplicates else "missing"
        notes.append(
            "listed more than once in PatientIndex.xlsx with different values"
            if status == "duplicate"
            else "not listed in PatientIndex.xlsx"
        )
        sex, age = details.get("dicom_sex"), details.get("dicom_age")
        row.update(
            index_status=status,
            clinical_source="dicom" if (sex or age is not None) else "none",
            sex=sex,
            age=age,
            symptoms_known=False,
            **{column: None for column in SYMPTOM_COLUMNS},
        )
        if sex or age is not None:
            notes.append("age and sex taken from the DICOM header; symptoms unknown")
    row["age_band"] = age_band(row["age"])
    row["manufacturer_group"] = manufacturer_group(row.get("manufacturer"))
    row["notes"] = "; ".join(notes)
    return row


def build_manifest(
    data: Path | str,
    out: Path,
    *,
    patient_index: Path | str | None = None,
    workers: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    started = time.perf_counter()
    data = Path(data).expanduser().resolve()
    with open_source(data) as source:
        layout = source.layout()
        if not layout.cases:
            raise DatasetError(
                f"No patient folders (TB/TB_xxx or NTM/Case_xxx) were found in {data}."
            )
        if patient_index is None:
            name = source.find(INDEX_FILE_NAME)
            if name is None:
                raise DatasetError(
                    f"{INDEX_FILE_NAME} was not found in {data}; pass it with --patient-index."
                )
            with source.open(name) as handle:
                index = read_patient_index(handle)
            index_location = f"{data.name}:{name}"
        else:
            index = read_patient_index(Path(patient_index).expanduser())
            index_location = str(patient_index)
        synthetic = source.is_synthetic
        kind = source.kind

    log(
        f"Found {len(layout.cases)} patient folders "
        f"({sum(c.label_name == 'TB' for c in layout.cases)} TB, "
        f"{sum(c.label_name == 'NTM' for c in layout.cases)} NTM) and "
        f"{len(index.records) + len(index.duplicates)} patients in {INDEX_FILE_NAME}."
    )
    workers = max(1, workers or min(8, os.cpu_count() or 1))
    details: dict[str, dict[str, Any]] = {}
    log(f"Reading DICOM headers with {workers} worker(s)…")
    results = run_parallel(
        _worker_details,
        layout.cases,
        workers=workers,
        initializer=_init_worker,
        initargs=(str(data),),
    )
    for done, (case_id, info) in enumerate(results, start=1):
        details[case_id] = info
        if done % 50 == 0 or done == len(layout.cases):
            log(f"  {done}/{len(layout.cases)} patients read")

    rows = [manifest_row(case, details[case.case_id], index) for case in layout.cases]
    sha = write_manifest(rows, out)
    folders = {c.case_id for c in layout.cases}
    listed = set(index.records) | set(index.duplicates)
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "toolkit_version": __version__,
        "dataset": {"path": str(data), "kind": kind, "synthetic": synthetic},
        "patient_index": index_location,
        "manifest_sha256": sha,
        "counts": {
            "patients": len(rows),
            "tb": sum(r["label"] == 1 for r in rows),
            "ntm": sum(r["label"] == 0 for r in rows),
            "no_dicom_images": sum(not r.get("slices") for r in rows),
            "index_ok": sum(r["index_status"] == "ok" for r in rows),
            "index_duplicate": sum(r["index_status"] == "duplicate" for r in rows),
            "index_missing": sum(r["index_status"] == "missing" for r in rows),
        },
        "problems": {
            "layout": layout.problems,
            "patient_index": index.problems,
            "ignored_index_rows": index.ignored_rows,
            "listed_without_scan_folder": sorted(listed - folders),
            "duplicate_index_rows": {
                case_id: [
                    {"row": r.row, "sex": r.sex, "age": r.age, "symptoms": r.symptoms}
                    for r in found
                ]
                for case_id, found in index.duplicates.items()
            },
        },
        "seconds": round(time.perf_counter() - started, 1),
    }
    summary_path(out).write_text(json.dumps(summary, indent=2) + "\n")
    counts = summary["counts"]
    log(
        f"Wrote {out} ({counts['patients']} patients: {counts['tb']} TB, {counts['ntm']} NTM) in "
        f"{summary['seconds']:.0f} s."
    )
    for label, items in (
        (
            "Not listed in PatientIndex.xlsx",
            [r["case_id"] for r in rows if r["index_status"] == "missing"],
        ),
        ("Listed twice with different values", sorted(index.duplicates)),
        ("Listed without a scan folder", summary["problems"]["listed_without_scan_folder"]),
        ("No DICOM images", [r["case_id"] for r in rows if not r.get("slices")]),
        ("Ignored index rows", index.ignored_rows),
        ("Folder problems", layout.problems),
    ):
        if items:
            log(f"  {label}: {', '.join(items[:20])}{' …' if len(items) > 20 else ''}")
    return summary
