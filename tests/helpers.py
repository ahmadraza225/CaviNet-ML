"""Synthetic CT volumes and DICOM series for the ml tests (no patient data)."""

from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def chest_volume(depth: int = 40, size: int = 64, lungs: bool = True) -> np.ndarray:
    """HU volume [z, y, x]: air, a soft-tissue body, two lungs that shrink at both ends,
    and scanner padding (-3024) in the corners."""
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, depth),
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        indexing="ij",
    )
    volume = np.full(z.shape, -1000.0, dtype=np.float32)
    volume[(x / 0.9) ** 2 + (y / 0.7) ** 2 <= 1] = 40.0
    if lungs:
        scale = np.sqrt(np.clip(1 - (z / 0.9) ** 2, 0, None))
        lung = ((np.abs(x) - 0.4) / (0.25 * scale + 1e-6)) ** 2 + (
            y / (0.45 * scale + 1e-6)
        ) ** 2 <= 1
        volume[lung & (scale > 0)] = -850.0
    volume[(np.abs(x) > 0.97) & (np.abs(y) > 0.97)] = -3024.0
    return volume


def image_from(volume: np.ndarray, spacing=(0.7, 0.7, 1.25)) -> sitk.Image:
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(spacing)
    return image


def write_series(
    directory: Path,
    volume: np.ndarray,
    spacing=(0.7, 0.7, 1.25),
    *,
    slope: float = 1.0,
    intercept: float = -1024.0,
    series_uid: str | None = None,
    shuffle_names: bool = False,
) -> list[Path]:
    """One DICOM file per slice; slice k is at z = k * spacing (feet first)."""
    directory.mkdir(parents=True, exist_ok=True)
    series_uid = series_uid or generate_uid()
    study_uid, frame_uid = generate_uid(), generate_uid()
    depth = volume.shape[0]
    order = np.random.default_rng(0).permutation(depth) if shuffle_names else np.arange(depth)
    paths = []
    for k in range(depth):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = "CT"
        ds.PatientName, ds.PatientID = "", ""
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, k * spacing[2]]
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.InstanceNumber = k + 1
        ds.Rows, ds.Columns = volume.shape[1:]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
        stored = np.round((volume[k] - intercept) / slope).astype(np.int16)
        ds.PixelData = stored.tobytes()
        path = directory / f"IM{order[k]:04d}.dcm"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def compress_jpeg_lossless(source: Path, target: Path) -> str:
    """Re-encode one slice as JPEG Lossless (1.2.840.10008.1.2.4.70) with GDCM."""
    image = sitk.ReadImage(str(source), sitk.sitkInt16)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetImageIO("GDCMImageIO")
    writer.SetUseCompression(True)
    writer.SetCompressor("JPEG")
    writer.SetFileName(str(target))
    writer.Execute(image)
    return str(pydicom.dcmread(target, stop_before_pixels=True).file_meta.TransferSyntaxUID)


def fake_segmenter(threshold: float = -500.0):
    """Stands in for lungmask: everything darker than `threshold` inside the body."""

    def segment(image: sitk.Image) -> np.ndarray:
        array = sitk.GetArrayFromImage(image)
        body = array > -950
        from scipy import ndimage

        filled = ndimage.binary_fill_holes(body)
        return ((array < threshold) & filled).astype(np.uint8)

    segment.calls = []  # type: ignore[attr-defined]
    return segment
