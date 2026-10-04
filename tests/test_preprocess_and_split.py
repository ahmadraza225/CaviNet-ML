"""FR-10.2 (preprocess: cache, QC, resume) and FR-10.3 (split: section 11.3)."""

import json
import os
from collections import Counter

import numpy as np
import pytest

from cavinet_ml.config import PreprocessingConfig
from cavinet_ml.dataset.cache import (
    QC_COLUMNS,
    cache_config,
    cache_file,
    check_cache_config,
    preprocess_dataset,
    read_qc,
)
from cavinet_ml.dataset.index import build_manifest
from cavinet_ml.dataset.source import DatasetError
from cavinet_ml.dataset.split import load_splits, make_splits, split_dataset, stratum
from cavinet_ml.dataset.synthetic import write_synthetic_dataset

QUIET = {"log": lambda message: None}
SMALL = PreprocessingConfig(output_size=(32, 32, 32))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("pre")
    data = write_synthetic_dataset(root, cases=9, broken=1)
    build_manifest(data, root / "work" / "manifest.csv", workers=1, **QUIET)
    return root


def _run(root, **kwargs):
    work = root / "work"
    options = {"use_lungmask": False, "config": SMALL, **QUIET, **kwargs}
    return preprocess_dataset(
        work / "manifest.csv", cache_dir=work / "cache", qc_path=work / "qc" / "qc.csv", **options
    )


def test_preprocess_caches_float16_volumes_and_writes_a_qc_report(dataset):
    summary = _run(dataset, workers=2, limit=4)
    assert summary.ok == 4 and summary.pending == 6 and not summary.failed
    work = dataset / "work"
    volume = np.load(cache_file(work / "cache", "TB_001"))
    assert volume.shape == (32, 32, 32) and volume.dtype == np.float16
    assert 0.0 <= float(volume.min()) and float(volume.max()) <= 1.0
    qc = read_qc(work / "qc" / "qc.csv")
    assert set(qc["TB_001"]) == set(QC_COLUMNS)
    row = qc["TB_001"]
    assert row["status"] == "ok" and row["used_fallback"] == "true"
    assert "Fallback crop" in row["warnings"]
    assert row["original_size"] == "96×96×56" and row["original_spacing_mm"] == "3.5×3.5×5"
    assert (work / "qc" / "qc_images" / "TB_001.png").is_file()
    assert not list((work / "tmp").iterdir())  # temporary patient folders are removed
    assert cache_config(work / "cache") == SMALL


def test_preprocess_resumes_and_lists_failures(dataset):
    summary = _run(dataset, workers=1)
    assert summary.skipped == 4  # done in the previous run
    assert summary.ok == 9 and summary.pending == 0
    assert list(summary.failed) == ["TB_007"]
    assert "No DICOM series" in summary.failed["TB_007"]
    again = _run(dataset, workers=1)
    assert again.skipped == 10 and again.ok == 9
    retried = _run(dataset, workers=1, retry_failed=True)
    assert retried.skipped == 9 and list(retried.failed) == ["TB_007"]
    qc = read_qc(dataset / "work" / "qc" / "qc.csv")
    assert len(qc) == 10  # one row per patient after deduplication


def test_preprocess_only_some_patients_and_refuses_unknown_ones(dataset):
    with pytest.raises(DatasetError, match="Not in the manifest"):
        _run(dataset, only=["TB_999"])


def test_cache_made_with_other_parameters_is_refused(tmp_path):
    check_cache_config(tmp_path / "cache", SMALL)
    check_cache_config(tmp_path / "cache", SMALL)  # same parameters: fine
    with pytest.raises(DatasetError, match="different preprocessing parameters"):
        check_cache_config(tmp_path / "cache", PreprocessingConfig())


def test_preprocess_with_the_real_lungmask(tmp_path):
    weights = os.environ.get("LUNGMASK_WEIGHTS")
    if not weights or not os.path.isfile(weights):
        pytest.skip("LUNGMASK_WEIGHTS not set (CI downloads the R231 weights)")
    data = write_synthetic_dataset(tmp_path, cases=4, broken=0)
    build_manifest(data, tmp_path / "work" / "manifest.csv", workers=1, **QUIET)
    summary = preprocess_dataset(
        tmp_path / "work" / "manifest.csv",
        cache_dir=tmp_path / "work" / "cache",
        qc_path=tmp_path / "work" / "qc.csv",
        lungmask_weights=weights,
        device="cpu",
        only=["TB_001"],
        config=SMALL,
        **QUIET,
    )
    row = read_qc(tmp_path / "work" / "qc.csv")["TB_001"]
    assert summary.ok == 1 and row["used_fallback"] == "false"
    assert float(row["lung_volume_ml"]) > 1000


# --- split ---------------------------------------------------------------------------------------


def population(n_tb=871, n_ntm=430, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tb + n_ntm):
        tb = i < n_tb
        rows.append(
            {
                "case_id": f"{'TB' if tb else 'Case'}_{i + 1:04d}",
                "label": int(tb),
                "sex": "male" if rng.random() < (0.72 if tb else 0.62) else "female",
                "age_band": rng.choice(["<40", "40-59", ">=60"]),
                "manufacturer_group": rng.choice(
                    ["GE", "TOSHIBA", "SIEMENS", "UIH"], p=[0.5, 0.3, 0.15, 0.05]
                ),
            }
        )
    return rows


def test_split_is_exactly_twenty_percent_stratified_and_deterministic():
    rows = population()
    test, folds = make_splits(rows)
    assert len(test) == 260  # round(0.2 × 1301)
    assert sorted(len(f) for f in folds) == [208, 208, 208, 208, 209]
    every = test + [c for f in folds for c in f]
    assert len(every) == len(set(every)) == len(rows)
    label = {r["case_id"]: r["label"] for r in rows}
    share = np.mean([label[c] for c in test])
    assert abs(share - 871 / 1301) < 0.01
    in_test = Counter(stratum(r) for r in rows if r["case_id"] in set(test))
    for key, n in Counter(stratum(r) for r in rows).items():
        assert abs(in_test[key] - 0.2 * n) < 1  # every stratum gives its share
    for fold in folds:
        assert abs(np.mean([label[c] for c in fold]) - (871 - share * 260) / 1041) < 0.02
    assert make_splits(rows) == (test, folds)
    assert make_splits(rows, seed=7) != (test, folds)


def test_split_parameters_are_checked():
    with pytest.raises(ValueError):
        make_splits(population(10, 5), test_fraction=1.5)
    with pytest.raises(ValueError):
        make_splits(population(10, 5), n_folds=1)
    with pytest.raises(DatasetError, match="Too few patients"):
        make_splits(population(2, 1), n_folds=5)


def test_split_dataset_excludes_failed_and_unlisted_patients_and_locks_the_file(dataset):
    work = dataset / "work"
    out = dataset / "splits.json"
    splits = split_dataset(work / "manifest.csv", work / "qc" / "qc.csv", out, n_folds=2, **QUIET)
    assert splits.excluded == {
        "TB_007": "preprocessing failed: No DICOM series was found in the stored scan.",
        "Case_002": "listed more than once in PatientIndex.xlsx with different values",
        "Case_003": "not listed in PatientIndex.xlsx",
    }
    assert len(splits.test) == 1 and splits.n_folds == 2 and len(splits.dev) == 6
    assert set(splits.train_ids(0)) == set(splits.folds[1])
    assert splits.fold_of(splits.folds[1][0]) == 1 and splits.fold_of(splits.test[0]) is None
    document = json.loads(out.read_text())
    assert document["seed"] == 42 and document["stratify_by"] == [
        "label",
        "sex",
        "age_band",
        "manufacturer_group",
    ]
    assert "TB_001" in json.dumps(document) and "manifest_sha256" in document
    with pytest.raises(DatasetError, match="locked"):
        split_dataset(work / "manifest.csv", work / "qc" / "qc.csv", out, **QUIET)


def test_split_needs_every_patient_preprocessed(tmp_path, dataset):
    work = dataset / "work"
    qc = tmp_path / "qc.csv"
    qc.write_text(",".join(QC_COLUMNS) + "\nTB_001,ok\n")
    with pytest.raises(DatasetError, match="not preprocessed yet"):
        split_dataset(work / "manifest.csv", qc, tmp_path / "s.json", **QUIET)
    with pytest.raises(DatasetError, match="run `cavinet-ml preprocess` first"):
        split_dataset(work / "manifest.csv", tmp_path / "none.csv", tmp_path / "s.json", **QUIET)


def test_load_splits_rejects_overlaps(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "format": 1,
                "seed": 42,
                "test_fraction": 0.2,
                "test": ["TB_001"],
                "folds": [["TB_001"], ["TB_002"]],
            }
        )
    )
    with pytest.raises(DatasetError, match="more than one split"):
        load_splits(path)
    with pytest.raises(DatasetError, match="run `cavinet-ml split` first"):
        load_splits(tmp_path / "missing.json")
