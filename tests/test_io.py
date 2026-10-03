"""Section 11.1 step 1: DICOM series loading with SimpleITK."""

import numpy as np
import pytest
import SimpleITK as sitk
from helpers import chest_volume, compress_jpeg_lossless, write_series

from cavinet_ml.io import DicomLoadError, load_series


def test_loads_the_series_in_hounsfield_units_as_float32(tmp_path):
    volume = chest_volume(depth=12)
    write_series(tmp_path, volume, intercept=-1024)
    image = load_series(tmp_path)
    assert image.GetPixelID() == sitk.sitkFloat32
    np.testing.assert_allclose(sitk.GetArrayFromImage(image), volume)


def test_rescale_slope_and_intercept_are_applied(tmp_path):
    volume = chest_volume(depth=8)
    write_series(tmp_path, volume, slope=2.0, intercept=-2048)
    np.testing.assert_allclose(sitk.GetArrayFromImage(load_series(tmp_path)), volume, atol=1)


def test_slices_are_sorted_by_position_not_file_name(tmp_path):
    volume = chest_volume(depth=10)
    volume[np.arange(10), 0, 0] = np.arange(10) * 10  # a per-slice marker
    write_series(tmp_path, volume, shuffle_names=True)
    array = sitk.GetArrayFromImage(load_series(tmp_path))
    assert list(array[:, 0, 0]) == [k * 10 for k in range(10)]


def test_spacing_and_lps_orientation(tmp_path):
    write_series(tmp_path, chest_volume(depth=6), spacing=(0.8, 0.8, 2.5))
    image = load_series(tmp_path)
    assert image.GetSpacing() == pytest.approx((0.8, 0.8, 2.5))
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines
    assert orientation(image.GetDirection()) == "LPS"
    assert image.GetSize() == (64, 64, 6)


def test_jpeg_lossless_series_is_decoded(tmp_path):
    volume = chest_volume(depth=8)
    plain = write_series(tmp_path / "plain", volume)
    (tmp_path / "jpeg").mkdir()
    syntaxes = {compress_jpeg_lossless(path, tmp_path / "jpeg" / path.name) for path in plain}
    assert syntaxes == {"1.2.840.10008.1.2.4.70"}  # JPEG Lossless, SV1
    np.testing.assert_allclose(sitk.GetArrayFromImage(load_series(tmp_path / "jpeg")), volume)


def test_the_largest_series_is_read(tmp_path):
    write_series(tmp_path, chest_volume(depth=5), series_uid="1.2.3.1")
    write_series(tmp_path / "x", chest_volume(depth=9), series_uid="1.2.3.2")
    for path in (tmp_path / "x").iterdir():
        path.rename(tmp_path / f"b-{path.name}")
    assert load_series(tmp_path, strict=False).GetDepth() == 9
    with pytest.raises(DicomLoadError, match="5 of the 14 stored slices could not be read"):
        load_series(tmp_path)


def test_empty_or_damaged_series_is_a_clear_error(tmp_path):
    with pytest.raises(DicomLoadError, match="No DICOM series"):
        load_series(tmp_path)
    paths = write_series(tmp_path, chest_volume(depth=4))
    data = paths[2].read_bytes()
    paths[2].write_bytes(data[: len(data) - 3000])
    with pytest.raises(DicomLoadError, match="1 of the 4 stored slices could not be read"):
        load_series(tmp_path)
