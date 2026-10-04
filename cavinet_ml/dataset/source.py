"""Read the Kaggle dataset `damianhan/dicom-dataset` from its zip or from an extracted folder.

Layout (checked against the Kaggle file listing):

    PatientIndex.xlsx
    TB/TB_001/TB_001_0001.dcm …      871 TB patients
    NTM/Case_001/Case_001_0001.dcm …  430 NTM patients

Patient folders are found by name wherever they sit (a top-level folder added by unzipping is
fine): `TB_<n>` inside a `TB` folder, `Case_<n>` inside an `NTM` folder. The zip is never
extracted as a whole: `extract_case` copies one patient's files at a time (FR-10.2).
"""

import os
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO

INDEX_FILE_NAME = "PatientIndex.xlsx"
# The synthetic generator writes this file; the Kaggle dataset has none.
SYNTHETIC_MARKER = "SYNTHETIC_DATASET.txt"

LABELS = {"TB": 1, "NTM": 0}
_CASE_NAME = re.compile(r"^(?P<prefix>TB|Case)_?(?P<number>\d+)$", re.IGNORECASE)
_PREFIX_OF_CLASS = {"TB": "TB", "NTM": "Case"}
_IGNORED_PARTS = {"__MACOSX"}


class DatasetError(Exception):
    """The dataset cannot be read as the Kaggle layout; the message says why."""


def canonical_case_id(name: str) -> str | None:
    """`TB_001` / `Case_001` for a folder or index name such as `tb_1` or `Case_001`."""
    match = _CASE_NAME.match(name.strip())
    if not match:
        return None
    prefix = "TB" if match["prefix"].upper() == "TB" else "Case"
    return f"{prefix}_{int(match['number']):03d}"


def class_of_case_id(case_id: str) -> str:
    return "TB" if case_id.startswith("TB_") else "NTM"


@dataclass(frozen=True)
class CaseEntry:
    case_id: str  # TB_001 / Case_001
    label_name: str  # TB or NTM, from the class folder
    location: str  # the patient folder, relative to the dataset root (POSIX separators)
    files: tuple[str, ...] = field(repr=False)  # zip member names or paths relative to root

    @property
    def label(self) -> int:
        return LABELS[self.label_name]


@dataclass
class Layout:
    cases: list[CaseEntry]
    problems: list[str]  # folders that look like patients but break the naming rules


def _case_of(parts: tuple[str, ...]) -> tuple[str, str, int] | None:
    """(case_id, class name, index of the patient folder) for a file path, if it lies in a
    patient folder."""
    for i in range(1, len(parts) - 1):  # the last part is the file name
        case_id = canonical_case_id(parts[i])
        parent = parts[i - 1].upper()
        if case_id and parent in LABELS:
            return case_id, parent, i
    return None


def _layout(paths: list[str]) -> Layout:
    by_folder: dict[str, tuple[str, str, list[str]]] = {}
    for path in paths:
        parts = PurePosixPath(path).parts
        if not parts or any(p in _IGNORED_PARTS or p.startswith(".") for p in parts):
            continue
        found = _case_of(parts)
        if found is None:
            continue
        case_id, class_name, i = found
        folder = "/".join(parts[: i + 1])
        by_folder.setdefault(folder, (case_id, class_name, []))[2].append(path)

    problems: list[str] = []
    folders_of: dict[str, list[str]] = {}
    for folder, (case_id, class_name, _) in by_folder.items():
        if not case_id.startswith(_PREFIX_OF_CLASS[class_name] + "_"):
            problems.append(f"{folder}: a {case_id} folder inside the {class_name} folder; skipped")
            continue
        folders_of.setdefault(case_id, []).append(folder)
    cases = []
    for case_id, folders in folders_of.items():
        if len(folders) > 1:
            problems.append(
                f"{case_id} is in {len(folders)} folders ({', '.join(sorted(folders))}); skipped"
            )
            continue
        _, class_name, files = by_folder[folders[0]]
        cases.append(CaseEntry(case_id, class_name, folders[0], tuple(sorted(files))))
    return Layout(sorted(cases, key=_sort_key), sorted(problems))


def _sort_key(case: CaseEntry) -> tuple[int, int]:
    return (0 if case.label_name == "TB" else 1, int(case.case_id.split("_")[1]))


class DatasetSource:
    """The dataset as a zip file or a folder. Use `open_source(path)`."""

    kind: str
    path: Path

    def names(self) -> list[str]:
        raise NotImplementedError

    def open(self, name: str) -> BinaryIO:
        raise NotImplementedError

    def close(self) -> None:
        """Release file handles (the zip)."""

    def layout(self) -> Layout:
        return _layout(self.names())

    def find(self, file_name: str) -> str | None:
        """The shallowest file called `file_name` (case-insensitive), e.g. PatientIndex.xlsx."""
        matches = [n for n in self.names() if PurePosixPath(n).name.lower() == file_name.lower()]
        return min(matches, key=lambda n: (n.count("/"), n)) if matches else None

    @property
    def is_synthetic(self) -> bool:
        return self.find(SYNTHETIC_MARKER) is not None

    def extract_case(self, case: CaseEntry, destination: Path) -> Path:
        """Copy one patient's files into `destination` (flat; names made unique)."""
        destination.mkdir(parents=True, exist_ok=True)
        for number, name in enumerate(case.files):
            target = destination / f"{number:05d}_{PurePosixPath(name).name}"
            with self.open(name) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
        return destination

    def __enter__(self) -> "DatasetSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ZipSource(DatasetSource):
    kind = "zip"

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._zip = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as error:
            raise DatasetError(f"{path} is not a readable zip file ({error}).") from error
        self._names = [i.filename for i in self._zip.infolist() if not i.is_dir()]

    def names(self) -> list[str]:
        return self._names

    def open(self, name: str) -> BinaryIO:
        return self._zip.open(name)  # type: ignore[return-value]

    def close(self) -> None:
        self._zip.close()


class FolderSource(DatasetSource):
    kind = "folder"

    def __init__(self, path: Path) -> None:
        self.path = path
        names = []
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in _IGNORED_PARTS)
            relative = Path(root).relative_to(path)
            for name in sorted(files):
                if not name.startswith("."):
                    names.append((relative / name).as_posix())
        self._names = names

    def names(self) -> list[str]:
        return self._names

    def open(self, name: str) -> BinaryIO:
        return (self.path / name).open("rb")


def open_source(path: Path | str) -> DatasetSource:
    path = Path(path).expanduser()
    if path.is_dir():
        return FolderSource(path)
    if path.is_file():
        return ZipSource(path)
    raise DatasetError(f"{path} does not exist (expected the dataset zip or its folder).")
