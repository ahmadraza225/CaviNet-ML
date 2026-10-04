"""FR-10.1: reading the Kaggle layout (zip or folder) and PatientIndex.xlsx."""

import io
import zipfile

import pytest
from openpyxl import Workbook

from cavinet_ml.dataset.index import build_manifest, scan_details
from cavinet_ml.dataset.manifest import (
    MANIFEST_COLUMNS,
    age_band,
    manufacturer_group,
    read_manifest,
    read_summary,
)
from cavinet_ml.dataset.patient_index import SYMPTOM_COLUMNS, SYMPTOMS, read_patient_index
from cavinet_ml.dataset.source import (
    DatasetError,
    canonical_case_id,
    open_source,
)
from cavinet_ml.dataset.synthetic import write_synthetic_dataset

HEADER = ["Number", "Gender", "Age", *SYMPTOMS, "Annotation Status"]


def workbook(tb_rows, ntm_rows, *, title_rows=0, legend=True) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in (("NTM", ntm_rows), ("TB", tb_rows)):
        ws = wb.create_sheet(name)
        for _ in range(title_rows):
            ws.append(["Patient index"])
        ws.append(HEADER + (["Strain Type"] if name == "NTM" else []))
        for row in rows:
            ws.append(row)
        if name == "TB" and legend:
            ws.append([None] * 5)
            ws.append(["Gender    1:Male  2:Female"])
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


def row(case_id, sex=1, age=50, symptoms=(), extra=()):
    values = [1 if i in symptoms else None for i in range(12)]
    return [case_id, sex, age, *values, None, *extra]


# --- source -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [("TB_001", "TB_001"), ("tb_7", "TB_007"), ("Case_042", "Case_042"), ("case42", "Case_042")],
)
def test_case_ids_are_canonical(name, expected):
    assert canonical_case_id(name) == expected


def test_non_case_names_are_not_case_ids():
    assert canonical_case_id("Gender 1:Male 2:Female") is None
    assert canonical_case_id("TB") is None


def _zip(tmp_path, names):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"x")
    return path


def test_layout_finds_patients_under_any_top_folder(tmp_path):
    path = _zip(
        tmp_path,
        [
            "dicom-dataset/TB/TB_001/TB_001_0001.dcm",
            "dicom-dataset/TB/TB_001/TB_001_0002.dcm",
            "dicom-dataset/NTM/Case_001/sub/Case_001_0001.dcm",
            "dicom-dataset/PatientIndex.xlsx",
            "__MACOSX/TB/TB_002/._x.dcm",
            "dicom-dataset/TB/.hidden/x.dcm",
        ],
    )
    with open_source(path) as source:
        layout = source.layout()
        assert source.kind == "zip"
        assert source.find("patientindex.xlsx") == "dicom-dataset/PatientIndex.xlsx"
        assert not source.is_synthetic
    assert [(c.case_id, c.label_name, c.label, len(c.files)) for c in layout.cases] == [
        ("TB_001", "TB", 1, 2),
        ("Case_001", "NTM", 0, 1),
    ]
    assert layout.cases[0].location == "dicom-dataset/TB/TB_001"
    assert layout.problems == []


def test_layout_reports_misplaced_and_duplicated_folders(tmp_path):
    path = _zip(
        tmp_path,
        [
            "NTM/TB_005/a.dcm",  # a TB folder inside NTM
            "a/TB/TB_003/a.dcm",
            "b/TB/TB_003/a.dcm",  # the same patient twice
            "TB/TB_004/a.dcm",
        ],
    )
    with open_source(path) as source:
        layout = source.layout()
    assert [c.case_id for c in layout.cases] == ["TB_004"]
    assert any("TB_005 folder inside the NTM folder" in p for p in layout.problems)
    assert any("TB_003 is in 2 folders" in p for p in layout.problems)


def test_folder_and_zip_sources_see_the_same_patients(tmp_path):
    zipped = write_synthetic_dataset(tmp_path / "z", cases=4, broken=0)
    folder = write_synthetic_dataset(tmp_path / "f", cases=4, broken=0, as_zip=False)
    with open_source(zipped) as a, open_source(folder) as b:
        assert a.is_synthetic and b.is_synthetic
        assert [c.case_id for c in a.layout().cases] == [c.case_id for c in b.layout().cases]
        case = a.layout().cases[0]
        out = a.extract_case(case, tmp_path / "one")
        assert len(list(out.iterdir())) == len(case.files)


def test_missing_or_invalid_dataset_is_explained(tmp_path):
    with pytest.raises(DatasetError, match="does not exist"):
        open_source(tmp_path / "nothing.zip")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(DatasetError, match="not a readable zip"):
        open_source(bad)


# --- PatientIndex.xlsx ---------------------------------------------------------------------------


def test_index_rows_symptoms_and_legend():
    index = read_patient_index(
        workbook(
            [row("TB_001", 2, 15, symptoms=(1, 2)), row("TB_002", 1, 25)],
            [row("Case_001", 1, 60, symptoms=(0,), extra=("Mycobacterium avium",))],
        )
    )
    assert set(index.records) == {"TB_001", "TB_002", "Case_001"}
    tb1 = index.records["TB_001"]
    assert (tb1.sex, tb1.age, tb1.sheet) == ("female", 15, "TB")
    assert tb1.symptoms["cough"] == 1 and tb1.symptoms["expectoration"] == 1
    assert tb1.symptoms["chest_pain"] == 0  # blank = absent
    assert set(tb1.symptoms) == set(SYMPTOM_COLUMNS)
    assert index.records["Case_001"].symptoms["chest_pain"] == 1
    assert index.ignored_rows == ["TB row 5: Gender 1:Male 2:Female"]
    assert index.problems == []


def test_a_cell_holding_a_space_counts_as_absent():
    tb = row("TB_001")
    tb[3 + SYMPTOMS.index("Haemoptysis")] = " "
    index = read_patient_index(workbook([tb], []))
    assert index.records["TB_001"].symptoms["haemoptysis"] == 0
    assert index.problems == []


def test_header_below_title_rows_and_numeric_numbers():
    index = read_patient_index(
        workbook([["TB_1", 1, 40.0] + [1.0] + [None] * 12], [], title_rows=2)
    )
    record = index.records["TB_001"]
    assert record.age == 40 and record.symptoms["chest_pain"] == 1


def test_duplicates_with_different_values_are_not_guessed():
    index = read_patient_index(
        workbook([row("TB_001")], [row("Case_252", age=38), row("Case_252", age=71), row("Case_7")])
    )
    assert "Case_252" not in index.records
    assert [r.age for r in index.duplicates["Case_252"]] == [38, 71]
    assert any("Case_252 is listed 2 times with different values" in p for p in index.problems)


def test_identical_duplicates_are_kept_once():
    index = read_patient_index(workbook([row("TB_001"), row("TB_001")], []))
    assert "TB_001" in index.records and not index.duplicates
    assert any("identically" in p for p in index.problems)


def test_patient_in_the_wrong_sheet_and_bad_values_are_reported():
    index = read_patient_index(
        workbook([row("Case_009"), ["TB_002", 3, "unknown", 2] + [None] * 11], [])
    )
    assert "Case_009" not in index.records
    record = index.records["TB_002"]
    assert record.sex is None and record.age is None and record.symptoms["chest_pain"] == 0
    joined = " ".join(index.problems)
    assert "Case_009 is listed in the TB sheet" in joined
    assert "Gender 3" in joined and "Age 'unknown'" in joined and "Chest pain 2" in joined


def test_missing_sheet_or_column_is_an_error(tmp_path):
    wb = Workbook()
    wb.active.title = "TB"
    wb.active.append(["Number", "Gender", "Age"])
    path = tmp_path / "index.xlsx"
    wb.save(path)
    with pytest.raises(DatasetError, match="no NTM sheet"):
        read_patient_index(path)
    wb.create_sheet("NTM").append(HEADER)
    wb.save(path)
    with pytest.raises(DatasetError, match="missing columns"):
        read_patient_index(path)
    with pytest.raises(DatasetError, match="cannot be read"):
        read_patient_index(io.BytesIO(b"not xlsx"))


# --- manifest ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "age, band", [(None, "unknown"), (39, "<40"), (40, "40-59"), (59, "40-59"), (60, ">=60")]
)
def test_age_bands(age, band):
    assert age_band(age) == band


@pytest.mark.parametrize(
    "name, group",
    [
        ("GE MEDICAL SYSTEMS", "GE"),
        ("Toshiba", "TOSHIBA"),
        ("SIEMENS", "SIEMENS"),
        ("Philips Medical Systems", "PHILIPS"),
        ("United Imaging Healthcare", "UIH"),
        ("", "unknown"),
        (None, "unknown"),
        ("Acme Scanners", "ACME"),
    ],
)
def test_manufacturer_groups(name, group):
    assert manufacturer_group(name) == group


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("indexed")
    data = write_synthetic_dataset(root, cases=9, broken=1)
    summary = build_manifest(data, root / "work" / "manifest.csv", workers=2, log=lambda m: None)
    return root, summary, {r["case_id"]: r for r in read_manifest(root / "work" / "manifest.csv")}


def test_manifest_has_every_patient_and_column(indexed):
    root, summary, rows = indexed
    assert len(rows) == 10 and summary["counts"]["tb"] == 7 and summary["counts"]["ntm"] == 3
    with (root / "work" / "manifest.csv").open(encoding="utf-8") as handle:
        assert handle.readline().strip().split(",") == list(MANIFEST_COLUMNS)
    assert read_summary(root / "work" / "manifest.csv")["dataset"]["synthetic"] is True
    tb = rows["TB_001"]
    assert tb["label"] == 1 and tb["label_name"] == "TB" and tb["index_status"] == "ok"
    assert tb["slices"] == 56 and tb["slice_spacing_mm"] == 5.0 and tb["series"] == 1
    assert tb["manufacturer_group"] == "GE" and tb["kernel"] == "LUNG" and tb["compressed"] is False
    assert tb["dicom_age"] == tb["age"] and tb["symptoms_known"] is True


def test_the_series_with_the_most_slices_is_described(indexed):
    _, _, rows = indexed
    assert rows["TB_002"]["series"] == 2 and rows["TB_002"]["slices"] == 70
    assert "the one with the most slices (70) is used" in rows["TB_002"]["notes"]


def test_duplicate_and_unlisted_patients_keep_blank_clinical_values(indexed):
    _, summary, rows = indexed
    assert rows["Case_002"]["index_status"] == "duplicate"
    assert rows["Case_003"]["index_status"] == "missing"
    for case_id in ("Case_002", "Case_003"):
        assert rows[case_id]["age"] is None and rows[case_id]["sex"] is None
        assert rows[case_id]["cough"] is None and rows[case_id]["symptoms_known"] is False
        assert rows[case_id]["age_band"] == "unknown"
    assert summary["counts"]["index_duplicate"] == 1 and summary["counts"]["index_missing"] == 1
    assert "Case_002" in summary["problems"]["duplicate_index_rows"]


def test_folders_without_dicom_images_are_flagged(indexed):
    _, summary, rows = indexed
    assert rows["TB_007"]["slices"] == 0
    assert "no DICOM images found" in rows["TB_007"]["notes"]
    assert summary["counts"]["no_dicom_images"] == 1
    assert summary["problems"]["ignored_index_rows"] == ["TB row 12: Gender 1:Male 2:Female"]


def test_index_without_patient_folders_or_index_file_is_an_error(tmp_path):
    empty = _zip(tmp_path, ["readme.txt"])
    with pytest.raises(DatasetError, match="No patient folders"):
        build_manifest(empty, tmp_path / "m.csv", workers=1, log=lambda m: None)
    no_index = _zip(tmp_path, ["TB/TB_001/a.dcm"])
    with pytest.raises(DatasetError, match="PatientIndex.xlsx was not found"):
        build_manifest(no_index, tmp_path / "m.csv", workers=1, log=lambda m: None)


def test_scan_details_notes_unusual_scans(tmp_path):
    from helpers import chest_volume, write_series

    from cavinet_ml.dataset.source import CaseEntry, FolderSource

    folder = tmp_path / "TB" / "TB_001"
    write_series(folder, chest_volume(depth=20, size=16), spacing=(0.7, 0.7, 6.0))
    (folder / "notes.txt").write_text("not dicom", encoding="utf-8")
    with FolderSource(tmp_path) as source:
        case = source.layout().cases[0]
        assert isinstance(case, CaseEntry)
        details = scan_details(source, case)
    notes = " ".join(details["notes"])
    assert details["slices"] == 20 and details["dicom_files"] == 20
    assert "1 of 21 files are not readable DICOM images" in notes
    assert "slice spacing 6 mm is more than 5 mm" in notes
    assert "only 20 slices" in notes
