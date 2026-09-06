from dataclasses import dataclass

import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window


@dataclass(frozen=True)
class SourceTile:
    rgb: np.ndarray
    valid_mask: np.ndarray


def read_rgb_and_valid_mask(
    dataset,
    window: Window,
    output_shape: tuple[int, int] | None = None,
) -> SourceTile:
    read_options = {"window": window}
    mask_options = {"window": window}
    if output_shape is not None:
        height, width = output_shape
        read_options.update(
            out_shape=(3, height, width),
            resampling=Resampling.bilinear,
        )
        mask_options.update(
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )

    if output_shape is None:
        masked_bands = dataset.read([1, 2, 3], masked=True, **read_options)
        bands = np.asarray(masked_bands.data)
        valid_mask = np.any(~np.ma.getmaskarray(masked_bands), axis=0).astype(
            np.uint8
        )
    else:
        bands = dataset.read([1, 2, 3], **read_options)
        source_mask = dataset.dataset_mask(**mask_options)
        valid_mask = (source_mask != 0).astype(np.uint8)
    return SourceTile(
        rgb=np.moveaxis(bands, 0, -1),
        valid_mask=valid_mask,
    )
