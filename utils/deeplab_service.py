import io
import hashlib
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock, Semaphore
from typing import Optional

import numpy as np
import torch
from PIL import Image

from deeplab import DeeplabV3
from utils.classification_contract import PIPELINE_VERSION
from utils.predict_large_image import SlidingWindowPredictor
from utils.utils import cvtColor

MODEL_CACHE_CAPACITY = 2


@dataclass(frozen=True)
class ModelCacheKey:
    weight_sha256: str
    pipeline_version: str
    device: str


@dataclass
class CachedModel:
    model: DeeplabV3
    active_references: int = 0


_model_cache: OrderedDict[ModelCacheKey, CachedModel] = OrderedDict()
_weight_digest_cache: dict[tuple[str, int, int], str] = {}
_cache_lock = RLock()
_inference_semaphore = Semaphore(1)


class ModelLoadError(RuntimeError):
    """The selected weight cannot be loaded by the supported runtime."""


def _load_model_unlocked(weight_file_path: str) -> DeeplabV3:
    try:
        return DeeplabV3(model_path=weight_file_path)
    except Exception as exc:
        raise ModelLoadError("selected model is incompatible") from exc


def _file_sha256(weight_file_path: str) -> str:
    path = Path(weight_file_path)
    if not path.is_file():
        raise FileNotFoundError(f"模型权重文件不存在: {path}")
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _cache_lock:
        cached = _weight_digest_cache.get(cache_key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    value = digest.hexdigest()
    with _cache_lock:
        _weight_digest_cache[cache_key] = value
    return value


def _evict_inactive_lru() -> None:
    while len(_model_cache) > MODEL_CACHE_CAPACITY:
        for key, entry in _model_cache.items():
            if entry.active_references == 0:
                del _model_cache[key]
                break
        else:
            return


@contextmanager
def _model_reference(
    weight_file_path: str,
    weight_sha256: str | None,
):
    digest = (weight_sha256 or _file_sha256(weight_file_path)).lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ModelCacheKey(digest, PIPELINE_VERSION, device)
    with _cache_lock:
        entry = _model_cache.get(key)
        if entry is None:
            entry = CachedModel(_load_model_unlocked(weight_file_path))
            _model_cache[key] = entry
        entry.active_references += 1
        _model_cache.move_to_end(key)
        _evict_inactive_lru()
    try:
        yield entry.model
    finally:
        with _cache_lock:
            entry.active_references -= 1
            _evict_inactive_lru()


def clear_model_cache() -> None:
    with _cache_lock:
        if any(entry.active_references for entry in _model_cache.values()):
            raise RuntimeError("仍有任务正在使用模型缓存")
        _model_cache.clear()
        _weight_digest_cache.clear()


@contextmanager
def model_tile_predictor(
    weight_file_path: str,
    *,
    weight_sha256: str,
):
    with _model_reference(weight_file_path, weight_sha256) as model:
        predictor = SlidingWindowPredictor(model)

        def predict_tile(tile: np.ndarray) -> np.ndarray:
            with _inference_semaphore:
                return predictor._predict_tile(tile)

        yield predict_tile


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
            with _inference_semaphore:
                result[y1:y2, x1:x2] = predictor._predict_tile(tile)

    return result


def predict_mask(
    pil_image: Image.Image,
    model_id: int,
    weight_file_path: str,
    *,
    weight_sha256: str | None = None,
) -> np.ndarray:
    """Infer a class-index mask using the selected model."""
    with _model_reference(weight_file_path, weight_sha256) as model:
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
    *,
    weight_sha256: str | None = None,
) -> bytes:
    """Infer a class mask and independently render it as an RGBA PNG."""
    with _model_reference(weight_file_path, weight_sha256) as model:
        class_mask = _predict_mask(pil_image, model)
        return render_mask_png(class_mask, model, classes)
