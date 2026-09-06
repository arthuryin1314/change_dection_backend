import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import rasterio
from rasterio.windows import Window


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.pixel_area import pixel_area_m2


def measure(path: Path, tile_size: int) -> tuple[int, float, float]:
    started = perf_counter()
    valid_pixels = 0
    area_m2 = 0.0
    with rasterio.open(path) as dataset:
        grid_shape = (dataset.height, dataset.width)
        for row_off in range(0, dataset.height, tile_size):
            height = min(tile_size, dataset.height - row_off)
            for col_off in range(0, dataset.width, tile_size):
                width = min(tile_size, dataset.width - col_off)
                window = Window(col_off, row_off, width, height)
                bands = dataset.read([1, 2, 3], window=window, masked=True)
                valid = np.any(~np.ma.getmaskarray(bands), axis=0)
                valid_pixels += int(np.count_nonzero(valid))
                areas = pixel_area_m2(
                    dataset.crs,
                    dataset.transform,
                    grid_shape,
                    window,
                )
                area_m2 += float(np.sum(areas, where=valid))
    return valid_pixels, area_m2, perf_counter() - started


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--tile-size", type=int, default=2048)
    args = parser.parse_args()
    pixels, area, seconds = measure(args.path, args.tile_size)
    print(
        f"valid_pixels={pixels} area_m2={area:.6f} "
        f"area_hectares={area / 10_000:.6f} seconds={seconds:.3f}"
    )
