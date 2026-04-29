import io
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deeplab import DeeplabV3
from utils.utils import cvtColor, preprocess_input, resize_image

_model: Optional[DeeplabV3] = None


def load_model() -> None:
    global _model
    _model = DeeplabV3()


def get_model() -> DeeplabV3:
    if _model is None:
        raise RuntimeError("DeepLab model not loaded. Call load_model() first.")
    return _model


def _predict_mask(pil_image: Image.Image) -> np.ndarray:
    """Run inference and return a class-index mask with shape (H, W)."""
    model = get_model()
    image = cvtColor(pil_image)
    h, w = np.array(image).shape[:2]

    image_data, nw, nh = resize_image(image, (model.input_shape[1], model.input_shape[0]))
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
    )

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if model.cuda:
            images = images.cuda()

        pr = model.net(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = pr[
            int((model.input_shape[0] - nh) // 2): int((model.input_shape[0] - nh) // 2 + nh),
            int((model.input_shape[1] - nw) // 2): int((model.input_shape[1] - nw) // 2 + nw),
        ]
        pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

    return pr


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
