import io
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from utils.classification_result_lock import classification_result_lock

PALETTE = np.array([(0,0,0),(0,0,255),(0,128,0),(128,128,128),(0,255,0),(255,0,0)], dtype=np.uint8)

def _normalize_joint(bands):
    values = bands[np.isfinite(bands)]
    if values.size == 0:
        raise ValueError("原始影像没有可显示的有限像素")
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(bands.shape, dtype=np.uint8)
    return np.clip(np.nan_to_num((bands - low) / (high - low) * 255, nan=0), 0, 255).astype(np.uint8)

def read_report_rgb(path, max_side=1600):
    with rasterio.open(path) as dataset:
        if dataset.count < 3:
            raise ValueError("原始影像少于三个波段")
        scale = min(1.0, max_side / max(dataset.width, dataset.height))
        height, width = max(1, round(dataset.height * scale)), max(1, round(dataset.width * scale))
        bands = dataset.read([1,2,3], out_shape=(3,height,width), resampling=rasterio.enums.Resampling.bilinear)
    rgb = np.moveaxis(_normalize_joint(bands.astype(np.float32)), 0, -1)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()

def read_report_classification(classes_path, valid_mask_path, max_side=1600):
    with rasterio.open(classes_path) as classes, rasterio.open(valid_mask_path) as valid:
        scale = min(1.0, max_side / max(classes.width, classes.height))
        height, width = max(1, round(classes.height * scale)), max(1, round(classes.width * scale))
        values = classes.read(1, out_shape=(height,width), resampling=rasterio.enums.Resampling.nearest)
        mask = valid.read(1, out_shape=(height,width), resampling=rasterio.enums.Resampling.nearest)
    if np.any(values > 5) or np.any((mask != 0) & (mask != 1)):
        raise ValueError("分类图数据不符合六类结果契约")
    rgb = np.where(mask[..., None] == 1, PALETTE[values], 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
    return output.getvalue()

def read_period_images(period, max_side=1600):
    with classification_result_lock(Path(period.classes_path).parent):
        return read_report_rgb(period.image_path, max_side), read_report_classification(period.classes_path, period.valid_mask_path, max_side)

