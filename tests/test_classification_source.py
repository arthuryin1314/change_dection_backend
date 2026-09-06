import numpy as np
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.windows import Window

from utils.classification_source import read_rgb_and_valid_mask


def _source_dataset(data):
    memory_file = MemoryFile()
    dataset = memory_file.open(
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine(0.01, 0, 110, 0, -0.01, 30),
        nodata=0,
    )
    dataset.write(data)
    return memory_file, dataset


def test_dataset_mask_uses_or_semantics_across_rgb_bands():
    data = np.array(
        [
            [[0, 0]],
            [[5, 0]],
            [[5, 0]],
        ],
        dtype=np.uint8,
    )
    memory_file, dataset = _source_dataset(data)
    try:
        tile = read_rgb_and_valid_mask(dataset, Window(0, 0, 2, 1))
    finally:
        dataset.close()
        memory_file.close()

    assert tile.rgb.tolist() == [[[0, 5, 5], [0, 0, 0]]]
    assert tile.valid_mask.tolist() == [[1, 0]]


def test_resampled_mask_is_nearest_neighbor_not_interpolated():
    data = np.array(
        [
            [[9, 0], [0, 0]],
            [[9, 0], [0, 0]],
            [[9, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    memory_file, dataset = _source_dataset(data)
    try:
        tile = read_rgb_and_valid_mask(
            dataset,
            Window(0, 0, 2, 2),
            output_shape=(4, 4),
        )
    finally:
        dataset.close()
        memory_file.close()

    assert tile.valid_mask.tolist() == [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert set(np.unique(tile.valid_mask)) == {0, 1}


def test_valid_black_pixel_limit_is_explicitly_source_mask_driven():
    data = np.full((3, 1, 1), 7, dtype=np.uint8)
    memory_file, dataset = _source_dataset(data)
    try:
        tile = read_rgb_and_valid_mask(dataset, Window(0, 0, 1, 1))
    finally:
        dataset.close()
        memory_file.close()

    tile.rgb[0, 0] = 0
    assert tile.valid_mask[0, 0] == 1


def test_native_read_gets_rgb_and_validity_without_a_second_dataset_mask_read():
    data = np.full((3, 2, 2), 7, dtype=np.uint8)
    memory_file, dataset = _source_dataset(data)

    class DatasetProxy:
        height = dataset.height
        width = dataset.width

        def read(self, *args, **kwargs):
            return dataset.read(*args, **kwargs)

        def dataset_mask(self, *args, **kwargs):
            raise AssertionError("native reads must not issue a second mask read")

    try:
        tile = read_rgb_and_valid_mask(DatasetProxy(), Window(0, 0, 2, 2))
    finally:
        dataset.close()
        memory_file.close()

    assert tile.rgb.shape == (2, 2, 3)
    assert np.all(tile.valid_mask == 1)
