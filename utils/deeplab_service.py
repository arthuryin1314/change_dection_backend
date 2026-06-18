import io
from typing import Optional

import numpy as np
from PIL import Image

from deeplab import DeeplabV3
from utils.predict_large_image import SlidingWindowPredictor
from utils.utils import cvtColor

_model: Optional[DeeplabV3] = None


def load_model() -> None:
    global _model
    _model = DeeplabV3()


def get_model() -> DeeplabV3:
    if _model is None:
        raise RuntimeError("DeepLab model not loaded. Call load_model() first.")
    return _model


def _predict_mask(pil_image: Image.Image) -> np.ndarray:
    """Run sliding-window inference and return a class-index mask with shape (H, W)."""
    model = get_model()
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


def segment_rgba_png(pil_image: Image.Image, classes: Optional[list[int]] = None) -> bytes:
    """Return segmentation as RGBA PNG bytes.

    classes: class IDs to render (1-5). None renders all foreground classes.
    Background (class 0) is always transparent.
    """
    model = get_model()
    pr = _predict_mask(pil_image)

    colors = np.array(model.colors, dtype=np.uint8)
    h, w = pr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = colors[pr]
    if classes is None:
        rgba[:, :, 3] = np.where(pr == 0, 0, 180)
    else:
        rgba[:, :, 3] = np.where(np.isin(pr, classes), 180, 0)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()
