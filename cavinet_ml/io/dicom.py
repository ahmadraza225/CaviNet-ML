"""Section 11.1 step 1: read a DICOM series in Hounsfield units.

SimpleITK's GDCM reader decodes compressed transfer syntaxes (JPEG-Lossless included),
sorts the slices by position and applies RescaleSlope/RescaleIntercept, so the voxel values
are Hounsfield units. The volume is reoriented to LPS so every scan has the same axes:
array index [z, y, x] runs feet→head, anterior→posterior and right→left.
"""

from pathlib import Path

import SimpleITK as sitk


class DicomLoadError(Exception):
    """The series cannot be read; the message is safe to show to a doctor."""


def load_series(directory: Path | str, *, strict: bool = True) -> sitk.Image:
    """The largest DICOM series in `directory` as a float32 HU image in LPS orientation.

    With `strict` (the application: a case folder holds exactly one stored series) every file
    in the folder must be part of that series, so a damaged slice cannot be skipped silently.
    The training toolkit reads dataset folders with `strict=False`.
    """
    folder = Path(directory)
    directory = str(directory)
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(directory)
    except RuntimeError:
        series_ids = ()
    if not series_ids:
        raise DicomLoadError("No DICOM series was found in the stored scan.")
    files_by_series = {
        series_id: reader.GetGDCMSeriesFileNames(directory, series_id) for series_id in series_ids
    }
    files = max(files_by_series.values(), key=len)
    if strict:
        present = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
        if len(present) != len(files):
            raise DicomLoadError(
                f"{len(present) - len(files)} of the {len(present)} stored slices could not be "
                "read (damaged or not part of the scan)."
            )
    reader.SetFileNames(files)
    reader.SetImageIO("GDCMImageIO")
    reader.SetOutputPixelType(sitk.sitkFloat32)  # after RescaleSlope/Intercept: HU
    try:
        image = reader.Execute()
    except RuntimeError as error:
        raise DicomLoadError(
            "The scan could not be decoded (the image data is damaged or uses an "
            "unsupported compression)."
        ) from error
    if image.GetDimension() != 3 or image.GetDepth() < 2:
        raise DicomLoadError("The stored scan is not a 3D series.")
    return sitk.DICOMOrient(image, "LPS")
