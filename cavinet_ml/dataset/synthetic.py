"""A small synthetic look-alike of the Kaggle dataset, for tests and rehearsals. No patient data.

It copies the layout and the known quirks of `damianhan/dicom-dataset` (section 7):

- `TB/TB_001/TB_001_0001.dcm`, `NTM/Case_001/…` and `PatientIndex.xlsx` (sheets NTM and TB,
  the real column names, 1 = present / blank = absent, a cell holding a single space, three
  blank rows and a legend row at the end of the TB sheet);
- one NTM number listed twice with different values and one left out, as in the real file;
- uncompressed DICOM slices whose file-name order differs from the slice order, blank
  PatientID/PatientName, sex and age in the header, padding of -3024 HU outside the scan
  circle, several scanner makes, kernels and slice spacings;
- one patient with an extra 3-slice series (the series with the most slices must be used) and
  `broken` patients whose folder holds no DICOM images.

"TB" scans carry thick-walled cavities in the upper lungs and "NTM" scans small nodules lower
down, so a model can learn something; the numbers mean nothing clinically. A marker file
(SYNTHETIC_DATASET.txt) makes the toolkit label everything built from it as a demo.
"""

import io
import zipfile
from pathlib import Path

import numpy as np

from cavinet_ml.dataset.patient_index import SYMPTOMS
from cavinet_ml.dataset.source import INDEX_FILE_NAME, SYNTHETIC_MARKER

# (manufacturer, model, kernel, slice spacing in mm)
SCANNERS = (
    ("GE MEDICAL SYSTEMS", "BrightSpeed", "LUNG", 5.0),
    ("TOSHIBA", "Aquilion ONE", "FC52", 4.0),
    ("SIEMENS", "SOMATOM Definition AS", "B70f", 5.0),
)
# Probability of each symptom (SYMPTOMS order) for TB and NTM, roughly like the real index.
_SYMPTOM_P = {
    "TB": (0.12, 0.87, 0.70, 0.47, 0.10, 0.15, 0.10, 0.01, 0.03, 0.13, 0.06, 0.11),
    "NTM": (0.13, 0.77, 0.68, 0.38, 0.10, 0.15, 0.16, 0.02, 0.01, 0.31, 0.10, 0.23),
}
_HEADER = ("Number", "Gender", "Age", *SYMPTOMS, "Annotation Status")
_LEGEND = "Gender    1:Male  2:Female"
_STRAINS = ("Mycobacterium intracellulare", "Mycobacterium abscessus", "Mycobacterium kansasii")
MATRIX = 96  # pixels per row/column
PIXEL_MM = 3.5  # 336 mm field of view
CHEST_MM = 280.0  # head-to-feet extent of each scan


def _volume(label: str, depth: int, rng: np.random.Generator) -> np.ndarray:
    """HU volume [z, y, x], z from feet to head."""
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, depth),
        np.linspace(-1, 1, MATRIX),
        np.linspace(-1, 1, MATRIX),
        indexing="ij",
    )
    hu = np.full(z.shape, -1000.0, dtype=np.float32)
    hu[(x / 0.85) ** 2 + (y / 0.62) ** 2 <= 1] = 40.0
    scale = np.sqrt(np.clip(1 - (z / 0.92) ** 2, 0, None))
    lungs = []
    for side in (-1, 1):
        cx = side * rng.uniform(0.36, 0.42)
        lung = ((x - cx) / (0.27 * scale + 1e-6)) ** 2 + ((y + 0.05) / (0.42 * scale + 1e-6)) ** 2
        hu[(lung <= 1) & (scale > 0)] = -850.0
        lungs.append(cx)

    def sphere(centre, radius):
        cz, cy, cx = centre
        return (z - cz) ** 2 * 0.25 + (y - cy) ** 2 + (x - cx) ** 2 <= radius**2

    if label == "TB":  # thick-walled cavities in the upper lungs
        for _ in range(rng.integers(1, 4)):
            centre = (rng.uniform(0.25, 0.6), rng.uniform(-0.2, 0.1), lungs[rng.integers(2)])
            outer = rng.uniform(0.09, 0.13)
            hu[sphere(centre, outer)] = 30.0
            hu[sphere(centre, outer * 0.55)] = -1000.0
    else:  # many small nodules lower down
        for _ in range(rng.integers(10, 20)):
            centre = (
                rng.uniform(-0.6, 0.2),
                rng.uniform(-0.3, 0.2),
                lungs[rng.integers(2)] + rng.uniform(-0.12, 0.12),
            )
            hu[sphere(centre, rng.uniform(0.025, 0.04))] = 40.0
    hu += rng.normal(0.0, 15.0, hu.shape).astype(np.float32)
    hu[x**2 + y**2 > 1.0] = -3024.0  # outside the scan circle
    return hu


def _slices(
    case_id: str,
    label: str,
    patient: dict,
    scanner: tuple[str, str, str, float],
    rng: np.random.Generator,
    extra_series: bool,
) -> list[tuple[str, bytes]]:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    manufacturer, model, kernel, spacing = scanner
    depth = int(round(CHEST_MM / spacing))
    volume = _volume(label, depth, rng)
    study, frame = generate_uid(), generate_uid()

    def encode(index: int, pixels: np.ndarray, series_uid: str, number: int, z_mm: float):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series_uid, frame
        ds.SeriesNumber, ds.InstanceNumber = number, index + 1
        ds.Modality = "CT"
        ds.PatientName, ds.PatientID = "", ""  # blank, as in the dataset
        ds.PatientSex = "M" if patient["sex"] == 1 else "F"
        ds.PatientAge = f"{patient['age']:03d}Y"
        ds.InstitutionName = "Synthetic Hospital"
        ds.Manufacturer, ds.ManufacturerModelName, ds.ConvolutionKernel = (
            manufacturer,
            model,
            kernel,
        )
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-MATRIX * PIXEL_MM / 2, -MATRIX * PIXEL_MM / 2, z_mm]
        ds.PixelSpacing = [PIXEL_MM, PIXEL_MM]
        ds.SliceThickness = spacing
        ds.Rows = ds.Columns = MATRIX
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = np.round(pixels + 1024).astype(np.int16).tobytes()
        buffer = io.BytesIO()
        ds.save_as(buffer, enforce_file_format=True)
        return buffer.getvalue()

    series_uid = generate_uid()
    images = [encode(k, volume[k], series_uid, 2, -CHEST_MM + k * spacing) for k in range(depth)]
    if extra_series:  # a short second series, e.g. a few reconstructed slices
        uid = generate_uid()
        images += [encode(k, volume[depth // 2], uid, 3, -CHEST_MM / 2 + k) for k in range(3)]
    order = rng.permutation(len(images)) + 1  # file names are not in slice order
    return [(f"{case_id}_{order[i]:04d}.dcm", data) for i, data in enumerate(images)]


def _patient(label: str, rng: np.random.Generator) -> dict:
    mean, sd, male = (45, 15, 0.72) if label == "TB" else (60, 12, 0.62)
    return {
        "sex": 1 if rng.random() < male else 2,
        "age": int(np.clip(round(rng.normal(mean, sd)), 15, 88)),
        "symptoms": [int(rng.random() < p) for p in _SYMPTOM_P[label]],
        "annotated": int(rng.random() < 0.2),
        "strain": _STRAINS[rng.integers(len(_STRAINS))],
    }


def _index_workbook(patients: dict[str, dict], quirks: bool) -> bytes:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    ntm_ids = sorted(c for c in patients if c.startswith("Case_"))
    for sheet in ("NTM", "TB"):
        ws = workbook.create_sheet(sheet)
        ws.append(_HEADER + (("Strain Type",) if sheet == "NTM" else ()))
        ids = ntm_ids if sheet == "NTM" else sorted(c for c in patients if c.startswith("TB_"))
        rows = []
        for case_id in ids:
            p = patients[case_id]
            row = [case_id, p["sex"], p["age"], *(1 if s else None for s in p["symptoms"])]
            row.append(1 if p["annotated"] else None)
            if sheet == "NTM":
                row.append(p["strain"])
            rows.append(row)
        if sheet == "NTM" and quirks and len(rows) >= 3:
            # As in the published file: one number twice (different values), one left out.
            twin = list(rows[1])
            twin[2] = int(twin[2]) + 17
            rows.insert(2, twin)
            del rows[3]  # the third NTM patient is no longer listed
        if sheet == "TB" and rows:
            rows[0][3 + SYMPTOMS.index("Haemoptysis")] = " "  # a blank-looking cell
        for row in rows:
            ws.append(row)
        if sheet == "TB":
            for _ in range(3):
                ws.append([None] * len(_HEADER))
            ws.append([_LEGEND])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def write_synthetic_dataset(
    out: Path | str,
    *,
    cases: int = 24,
    broken: int = 1,
    seed: int = 42,
    quirks: bool = True,
    as_zip: bool = True,
) -> Path:
    """Write `dicom-dataset.zip` (or a `dicom-dataset` folder) into `out`; returns its path.
    About two thirds of the `cases` are TB, like the real dataset."""
    if cases < 4:
        raise ValueError("at least 4 cases are needed")
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_tb = int(round(cases * 2 / 3))
    labels = {f"TB_{i:03d}": "TB" for i in range(1, n_tb + 1)}
    labels.update({f"Case_{i:03d}": "NTM" for i in range(1, cases - n_tb + 1)})
    patients = {case_id: _patient(label, rng) for case_id, label in labels.items()}
    broken_ids = [f"TB_{n_tb + i:03d}" for i in range(1, broken + 1)]
    for case_id in broken_ids:
        patients[case_id] = _patient("TB", rng)

    files: list[tuple[str, bytes]] = []
    for number, (case_id, label) in enumerate(labels.items()):
        scanner = SCANNERS[number % len(SCANNERS)]
        folder = f"{label}/{case_id}"
        for name, data in _slices(case_id, label, patients[case_id], scanner, rng, number == 1):
            files.append((f"{folder}/{name}", data))
    for case_id in broken_ids:
        for k in range(1, 4):
            files.append((f"TB/{case_id}/{case_id}_{k:04d}.dcm", b"not a DICOM image\n"))
    files.append((INDEX_FILE_NAME, _index_workbook(patients, quirks)))
    files.append(
        (
            SYNTHETIC_MARKER,
            b"Synthetic look-alike of the Kaggle dataset damianhan/dicom-dataset made by "
            b"cavinet_ml.dataset.synthetic. No patient data. Models trained on it are demos.\n",
        )
    )

    if as_zip:
        path = out / "dicom-dataset.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, data in files:
                archive.writestr(name, data)
        return path
    root = out / "dicom-dataset"
    for name, data in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root
