from unittest.mock import Mock

import numpy as np
import pytest
from affine import Affine
from rasterio.io import MemoryFile

from utils.streaming_classification import StreamingMetrics, stream_classification_bands


def _dataset(width, height, *, data=None):
    if data is None:
        y, x = np.indices((height, width))
        data = np.stack(
            [
                (x % 250) + 1,
                (y % 250) + 1,
                ((x + y) % 250) + 1,
            ]
        ).astype(np.uint8)
    memory_file = MemoryFile()
    dataset = memory_file.open(
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:4528",
        transform=Affine(0.8, 0, 500000, 0, -0.8, 3200000),
        nodata=0,
    )
    dataset.write(data)
    return memory_file, dataset, np.moveaxis(data, 0, -1)


def _tile_prediction(tile):
    local_y, local_x = np.indices(tile.shape[:2])
    return ((tile[:, :, 0] + local_y + 2 * local_x) % 6).astype(np.uint8)


def _reference_overwrite(rgb, tile_size=512, overlap=128):
    height, width = rgb.shape[:2]
    stride = tile_size - overlap
    result = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y2 = min(y + tile_size, height)
            x2 = min(x + tile_size, width)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)
            result[y1:y2, x1:x2] = _tile_prediction(rgb[y1:y2, x1:x2])
    return result


def _collect(dataset, metrics=None):
    classes = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
    valid = np.zeros_like(classes)
    for band in stream_classification_bands(
        dataset,
        _tile_prediction,
        metrics=metrics,
    ):
        row = int(band.window.row_off)
        height = int(band.window.height)
        classes[row : row + height] = band.classes
        valid[row : row + height] = band.valid_mask
    return classes, valid


@pytest.mark.parametrize("size", [511, 512, 513, 895])
def test_streaming_is_pixel_equal_to_existing_row_major_overwrite(size):
    memory_file, dataset, rgb = _dataset(size, size)
    try:
        actual, valid = _collect(dataset)
    finally:
        dataset.close()
        memory_file.close()

    assert np.array_equal(actual[valid == 1], _reference_overwrite(rgb)[valid == 1])
    assert np.all(valid == 1)


def test_all_invalid_tiles_skip_inference_and_emit_zero(tmp_path):
    data = np.zeros((3, 4, 4), dtype=np.uint8)
    memory_file, dataset, _ = _dataset(4, 4, data=data)
    predictor = Mock(return_value=np.full((4, 4), 5, dtype=np.uint8))
    metrics = StreamingMetrics()
    try:
        bands = list(
            stream_classification_bands(
                dataset,
                predictor,
                tile_size=4,
                overlap=1,
                metrics=metrics,
            )
        )
    finally:
        dataset.close()
        memory_file.close()

    predictor.assert_not_called()
    assert len(bands) == 1
    assert np.all(bands[0].classes == 0)
    assert np.all(bands[0].valid_mask == 0)
    assert metrics.skipped_tiles == 4
    assert metrics.effective_tiles == 0


def test_partially_valid_tile_is_inferred_but_invalid_pixels_are_normalized():
    data = np.zeros((3, 4, 4), dtype=np.uint8)
    data[:, 1, 1] = 9
    memory_file, dataset, _ = _dataset(4, 4, data=data)
    predictor = Mock(return_value=np.full((4, 4), 5, dtype=np.uint8))
    try:
        bands = list(
            stream_classification_bands(
                dataset,
                predictor,
                tile_size=4,
                overlap=1,
            )
        )
    finally:
        dataset.close()
        memory_file.close()

    assert predictor.call_count == 4
    assert bands[0].classes[1, 1] == 5
    assert np.count_nonzero(bands[0].classes) == 1
