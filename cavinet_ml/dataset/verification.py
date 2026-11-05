"""`cavinet-ml verification-set` (Phase 9): real test-set scans for checking the application.

Picks patients of the locked test set (5 TB and 5 NTM by default, with seed 42; the first one is
the scan with the slice count nearest 300, for the timing check), copies each patient's DICOM
files from the dataset into its own zip, and writes expected.csv with the toolkit's probability
for each. scripts/verify_model.py then uploads the zips through the application and compares.

The zips contain real (de-identified, public) patient scans: keep them out of git.
"""

import csv
import random
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from cavinet_ml.dataset.manifest import read_manifest
from cavinet_ml.dataset.source import DatasetError, open_source
from cavinet_ml.dataset.split import load_splits

TIMING_SLICES = 300
EXPECTED_COLUMNS = ["case_id", "label", "toolkit_probability", "slices", "zip"]


def read_test_predictions(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise DatasetError(f"{path} not found; `cavinet-ml evaluate --split test` writes it.")
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: float(row["ct_probability"]) for row in csv.DictReader(handle)}


def choose_cases(
    rows: dict[str, dict], test_ids: list[str], per_class: int, seed: int
) -> list[str]:
    """per_class TB and per_class NTM test patients; the first is the one nearest 300 slices."""
    timing = min(test_ids, key=lambda c: (abs((rows[c]["slices"] or 0) - TIMING_SLICES), c))
    rng = random.Random(seed)
    chosen = [timing]
    for label in (1, 0):
        pool = sorted(c for c in test_ids if rows[c]["label"] == label and c != timing)
        wanted = per_class - (1 if rows[timing]["label"] == label else 0)
        if len(pool) < wanted:
            raise DatasetError(f"The test set has too few {'TB' if label else 'NTM'} patients.")
        chosen += rng.sample(pool, wanted)
    return chosen


def write_verification_set(
    data: Path | str,
    *,
    manifest: Path,
    splits: Path,
    predictions: Path,
    out: Path,
    per_class: int = 5,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> Path:
    rows = {r["case_id"]: r for r in read_manifest(manifest)}
    test_ids = [c for c in load_splits(splits).test if c in rows]
    probabilities = read_test_predictions(predictions)
    chosen = choose_cases(rows, test_ids, per_class, seed)
    missing = [c for c in chosen if c not in probabilities]
    if missing:
        raise DatasetError(f"No toolkit prediction for {', '.join(missing)} in {predictions}.")
    out.mkdir(parents=True, exist_ok=True)
    expected = out / "expected.csv"
    with open_source(data) as source:
        entries = {c.case_id: c for c in source.layout().cases}
        with expected.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(EXPECTED_COLUMNS)
            for number, case_id in enumerate(chosen, 1):
                if case_id not in entries:
                    raise DatasetError(f"{case_id} is not in the dataset at {data}.")
                archive = out / f"{case_id}.zip"
                with tempfile.TemporaryDirectory() as folder:
                    files = source.extract_case(entries[case_id], Path(folder) / case_id)
                    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                        for path in sorted(files.iterdir()):
                            bundle.write(path, f"{case_id}/{path.name}")
                    shutil.rmtree(files, ignore_errors=True)
                writer.writerow(
                    [
                        case_id,
                        "TB" if rows[case_id]["label"] else "NTM",
                        f"{probabilities[case_id]:.6f}",
                        rows[case_id]["slices"],
                        archive.name,
                    ]
                )
                log(f"[{number}/{len(chosen)}] {archive}")
    log(f"Wrote {len(chosen)} scans and {expected}. They are real scans: never commit them.")
    return expected
