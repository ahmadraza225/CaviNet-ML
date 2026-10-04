"""`cavinet-ml split` (FR-10.3, section 11.3): the locked test set and the 5 folds.

- Locked test set: 20% of patients, stratified by label × sex × age band (<40, 40-59, ≥60) ×
  scanner manufacturer, seed 42. Each stratum gives its proportional share; the shares are
  rounded by largest remainder so the total is exactly 20%.
- The remaining 80%: 5 folds with the same stratification (each stratum is dealt across the
  folds in turn, so fold sizes differ by at most one).
- One scan per patient. Excluded, with the reason: patients whose preprocessing failed, and
  patients without exactly one row in PatientIndex.xlsx (their label cannot be cross-checked).
- splits.json holds case IDs only and is committed to the repository. Once written it is not
  replaced without --force: the test set must stay locked.
"""

import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cavinet_ml import __version__
from cavinet_ml.dataset.cache import read_qc
from cavinet_ml.dataset.manifest import UNKNOWN, file_sha256, read_manifest
from cavinet_ml.dataset.source import DatasetError
from cavinet_ml.provenance import git_commit

FORMAT = 1
STRATIFY_BY = ("label", "sex", "age_band", "manufacturer_group")


@dataclass
class Splits:
    test: list[str]
    folds: list[list[str]]
    seed: int = 42
    test_fraction: float = 0.2
    excluded: dict[str, str] = field(default_factory=dict)
    manifest_sha256: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def dev(self) -> list[str]:
        return [c for fold in self.folds for c in fold]

    def fold_of(self, case_id: str) -> int | None:
        for k, fold in enumerate(self.folds):
            if case_id in fold:
                return k
        return None

    def train_ids(self, fold: int) -> list[str]:
        return [c for k, ids in enumerate(self.folds) if k != fold for c in ids]

    def val_ids(self, fold: int) -> list[str]:
        return list(self.folds[fold])


def stratum(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(key) if row.get(key) not in (None, "") else UNKNOWN) for key in STRATIFY_BY
    )


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def make_splits(
    rows: list[dict[str, Any]], *, seed: int = 42, test_fraction: float = 0.2, n_folds: int = 5
) -> tuple[list[str], list[list[str]]]:
    """(test IDs, folds of development IDs), deterministic for a given seed."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if n_folds < 2:
        raise ValueError("at least 2 folds are needed")
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row["case_id"])
    keys = sorted(groups)
    shuffled = {k: rng.permutation(sorted(groups[k])).tolist() for k in keys}

    n_test = _round_half_up(test_fraction * len(rows))
    quota = {k: test_fraction * len(shuffled[k]) for k in keys}
    take = {k: math.floor(quota[k]) for k in keys}
    tiebreak = dict(zip(keys, rng.random(len(keys)), strict=True))
    by_remainder = sorted(keys, key=lambda k: (-(quota[k] - take[k]), tiebreak[k]))
    for k in by_remainder[: max(0, n_test - sum(take.values()))]:
        take[k] += 1

    test = [c for k in keys for c in shuffled[k][: take[k]]]
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    position = 0
    for k in keys:
        for case_id in shuffled[k][take[k] :]:
            folds[position % n_folds].append(case_id)
            position += 1
    if min(len(f) for f in folds) == 0:
        raise DatasetError("Too few patients for the requested number of folds.")
    return test, folds


def _counts(ids: list[str], label_of: dict[str, int]) -> dict[str, int]:
    return {
        "n": len(ids),
        "tb": sum(label_of[c] == 1 for c in ids),
        "ntm": sum(label_of[c] == 0 for c in ids),
    }


def split_dataset(
    manifest_path: Path,
    qc_path: Path,
    out: Path,
    *,
    seed: int = 42,
    test_fraction: float = 0.2,
    n_folds: int = 5,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Splits:
    if out.exists() and not force:
        raise DatasetError(
            f"{out} already exists. The test set is locked once created; use --force only if "
            "no model has been evaluated on it yet."
        )
    rows = read_manifest(manifest_path)
    qc = read_qc(qc_path)
    if not qc:
        raise DatasetError(f"{qc_path} not found; run `cavinet-ml preprocess` first.")
    pending = [r["case_id"] for r in rows if r["case_id"] not in qc]
    if pending:
        raise DatasetError(
            f"{len(pending)} patients are not preprocessed yet ({', '.join(pending[:10])}); "
            "finish `cavinet-ml preprocess` before splitting."
        )
    excluded: dict[str, str] = {}
    for r in rows:
        if r["index_status"] != "ok":
            # Section 7.3: the label is cross-checked with PatientIndex.xlsx; without one clear
            # row it cannot be, and the clinical baseline would lack this patient (H2 compares
            # both models on the same patients).
            excluded[r["case_id"]] = (
                "listed more than once in PatientIndex.xlsx with different values"
                if r["index_status"] == "duplicate"
                else "not listed in PatientIndex.xlsx"
            )
        elif qc[r["case_id"]]["status"] != "ok":
            excluded[r["case_id"]] = f"preprocessing failed: {qc[r['case_id']]['error']}"
    usable = [r for r in rows if r["case_id"] not in excluded]
    test, folds = make_splits(usable, seed=seed, test_fraction=test_fraction, n_folds=n_folds)

    label_of = {r["case_id"]: r["label"] for r in rows}
    strata: dict[str, dict[str, int]] = defaultdict(lambda: {"test": 0, "dev": 0})
    test_set = set(test)
    for row in usable:
        strata[stratum(row)]["test" if row["case_id"] in test_set else "dev"] += 1
    document = {
        "format": FORMAT,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "toolkit_version": __version__,
        "git_commit": git_commit(),
        "manifest_sha256": file_sha256(manifest_path),
        "seed": seed,
        "test_fraction": test_fraction,
        "n_folds": n_folds,
        "stratify_by": list(STRATIFY_BY),
        "age_bands": ["<40", "40-59", ">=60"],
        "counts": {
            "patients": len(rows),
            "usable": len(usable),
            "excluded": len(excluded),
            "test": _counts(test, label_of),
            "dev": _counts([c for f in folds for c in f], label_of),
            "folds": [_counts(f, label_of) for f in folds],
        },
        "strata": dict(sorted(strata.items())),
        "test": sorted(test, key=_order),
        "folds": [sorted(f, key=_order) for f in folds],
        "excluded": dict(sorted(excluded.items(), key=lambda kv: _order(kv[0]))),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    counts = document["counts"]
    log(
        f"Wrote {out}: test {counts['test']['n']} ({counts['test']['tb']} TB, "
        f"{counts['test']['ntm']} NTM), development {counts['dev']['n']} in {n_folds} folds of "
        f"{', '.join(str(f['n']) for f in counts['folds'])}; {len(excluded)} excluded."
    )
    return load_splits(out)


def _order(case_id: str) -> tuple[int, int]:
    prefix, number = case_id.split("_")
    return (0 if prefix == "TB" else 1, int(number))


def load_splits(path: Path) -> Splits:
    if not path.is_file():
        raise DatasetError(f"{path} not found; run `cavinet-ml split` first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise DatasetError(f"{path}: unsupported splits format {data.get('format')!r}")
    test, folds = list(data["test"]), [list(f) for f in data["folds"]]
    every = test + [c for f in folds for c in f]
    if len(every) != len(set(every)):
        raise DatasetError(f"{path}: a patient appears in more than one split")
    return Splits(
        test=test,
        folds=folds,
        seed=data["seed"],
        test_fraction=data["test_fraction"],
        excluded=data.get("excluded", {}),
        manifest_sha256=data.get("manifest_sha256"),
        raw=data,
    )
