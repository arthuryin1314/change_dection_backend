import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from utils.tif_reader import (
    NATIVE_PIXEL_CAP,
    NoOverlapError,
    UnsupportedSrsError,
    _cap_dimensions,
    read_tif_rgb_window,
)


def test_cap_dimensions_keeps_small_window():
    assert _cap_dimensions(4000, 4000) == (4000, 4000, False)


def test_cap_dimensions_scales_large_window_proportionally():
    width, height, capped = _cap_dimensions(20000, 10000)

    assert max(width, height) == NATIVE_PIXEL_CAP
    assert abs(width / height - 20000 / 10000) < 0.01
    assert capped is True


def test_cap_dimensions_keeps_boundary_value():
    assert _cap_dimensions(8192, 8192) == (8192, 8192, False)


@pytest.fixture()
def rgb_tif(tmp_path):
    tif_path = tmp_path / "rgb.tif"
    data = np.zeros((3, 100, 100), dtype=np.uint8)
    data[0, :, :] = 10
    data[1, :, :] = 20
    data[2, :, :] = 30

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 100, 1, 1),
    ) as ds:
        ds.write(data)

    return tif_path


def test_read_tif_rgb_window_returns_expected_size(rgb_tif):
    result = read_tif_rgb_window(rgb_tif, "10,80,20,90", "EPSG:4326")

    assert result.image.size == (10, 10)
    assert result.requested_native_width == 10
    assert result.requested_native_height == 10
    assert result.effective_offset == (0, 0)
    assert result.effective_size == (10, 10)


def test_read_tif_rgb_window_rejects_outside_bbox(rgb_tif):
    with pytest.raises(NoOverlapError):
        read_tif_rgb_window(rgb_tif, "200,200,210,210", "EPSG:4326")


def test_read_tif_rgb_window_rejects_unsupported_srs(rgb_tif):
    with pytest.raises(UnsupportedSrsError):
        read_tif_rgb_window(rgb_tif, "10,80,20,90", "EPSG:99999")
