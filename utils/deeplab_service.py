import io
from threading import RLock
from typing import Optional

import numpy as np
from PIL import Image

from deeplab import DeeplabV3
from utils.predict_large_image import SlidingWindowPredictor
from utils.utils import cvtColor

_model: Optional[DeeplabV3] = None
_model_key: Optional[tuple[int, str]] = None
# ponytail: one process-wide lock keeps the single-model cache consistent; shard by model/device if throughput requires it.
_model_lock = RLock()


class ModelLoadError(RuntimeError):
    """The selected weight cannot be loaded by the supported runtime."""


def _load_model_unlocked(model_id: int, weight_file_path: str) -> DeeplabV3:
    global _model, _model_key

    key = (model_id, weight_file_path)
    if _model is None or _model_key != key:
        try:
            _model = DeeplabV3(model_path=weight_file_path)
        except Exception as exc:
            raise ModelLoadError("selected model is incompatible") from exc
        _model_key = key
    return _model


def _predict_mask(pil_image: Image.Image, model: DeeplabV3) -> np.ndarray:
    """Run sliding-window inference and return a class-index mask with shape (H, W)."""
    predictor = SlidingWindowPredictor(model)

    image = cvtColor(pil_image)
    original_w, original_h = image.size
    image_np = np.array(image)

    result = np.zeros((original_h, original_w), dtype=np.uint8)

    for y in range(0, original_h, predictor.stride):
        for x in range(0, original_w, predictor.stride):
            y2 = min(y + predictor.tile_size, original_h)
            x2 = min(x + predictor.tile_size, original_w)
            y1 = max(0, y2 - predictor.tile_size)
            x1 = max(0, x2 - predictor.tile_size)
            tile = image_np[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = predictor._predict_tile(tile)

    return result


def predict_mask(pil_image: Image.Image, model_id: int, weight_file_path: str) -> np.ndarray:
    """Infer a class-index mask using the selected model."""
    with _model_lock:
        model = _load_model_unlocked(model_id, weight_file_path)
        return _predict_mask(pil_image, model)


def render_mask_png(
    class_mask: np.ndarray,
    model: DeeplabV3,
    classes: Optional[list[int]] = None,
) -> bytes:
    """Render an already-inferred class mask as an RGBA PNG."""
    colors = np.array(model.colors, dtype=np.uint8)
    h, w = class_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = colors[class_mask]
    if classes is None:
        rgba[:, :, 3] = np.where(class_mask == 0, 0, 180)
    else:
        rgba[:, :, 3] = np.where(np.isin(class_mask, classes), 180, 0)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def segment_rgba_png(
    pil_image: Image.Image,
    model_id: int,
    weight_file_path: str,
    classes: Optional[list[int]] = None,
) -> bytes:
    """Infer a class mask and independently render it as an RGBA PNG."""
    with _model_lock:
        model = _load_model_unlocked(model_id, weight_file_path)
        class_mask = _predict_mask(pil_image, model)
        return render_mask_png(class_mask, model, classes)
