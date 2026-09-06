import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


MODEL_ID = 11
WEIGHT_PATH = "weights/model.pth"
WEIGHT_SHA256 = "a" * 64


def _make_mock_model():
    model = MagicMock()
    model.colors = [
        (0, 0, 0),        # 0 background
        (0, 0, 255),      # 1 water
        (0, 128, 0),      # 2 woodland
        (128, 128, 128),  # 3 road
        (0, 255, 0),      # 4 cultivated land
        (255, 0, 0),      # 5 construction land
    ]
    return model


def test_segment_rgba_returns_bytes():
    """Return value must be bytes."""
    from utils import deeplab_service

    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=np.zeros((4, 4), dtype=int)):
        result = deeplab_service.segment_rgba_png(Image.new("RGB", (4, 4)), MODEL_ID, WEIGHT_PATH, weight_sha256=WEIGHT_SHA256)
        assert isinstance(result, bytes)


def test_segment_rgba_output_mode_is_rgba():
    """Output PNG must be RGBA."""
    from utils import deeplab_service

    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=np.zeros((4, 4), dtype=int)):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (4, 4)), MODEL_ID, WEIGHT_PATH, weight_sha256=WEIGHT_SHA256)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGBA"


def test_segment_rgba_background_alpha_is_zero():
    """Background class pixels must have alpha 0."""
    from utils import deeplab_service

    mask = np.array([[0, 0], [0, 0]])
    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)), MODEL_ID, WEIGHT_PATH, weight_sha256=WEIGHT_SHA256)
        img = Image.open(io.BytesIO(png_bytes))
        for pixel in img.getdata():
            assert pixel[3] == 0, f"background alpha should be 0, got {pixel[3]}"


def test_segment_rgba_foreground_alpha_is_180():
    """Foreground class pixels must have alpha 180."""
    from utils import deeplab_service

    mask = np.array([[1, 1], [1, 1]])
    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)), MODEL_ID, WEIGHT_PATH, weight_sha256=WEIGHT_SHA256)
        img = Image.open(io.BytesIO(png_bytes))
        for pixel in img.getdata():
            assert pixel[3] == 180, f"foreground alpha should be 180, got {pixel[3]}"
            assert pixel[:3] == (0, 0, 255), f"water color should be (0, 0, 255), got {pixel[:3]}"


def test_segment_rgba_mixed_classes():
    """Mixed background and foreground classes use the expected alpha values."""
    from utils import deeplab_service

    mask = np.array([[0, 1], [2, 3]])
    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)), MODEL_ID, WEIGHT_PATH, weight_sha256=WEIGHT_SHA256)
        img = Image.open(io.BytesIO(png_bytes))
        pixels = list(img.getdata())
        assert pixels[0][3] == 0
        assert pixels[1][3] == 180
        assert pixels[2][3] == 180
        assert pixels[3][3] == 180


def test_segment_rgba_classes_filter_hides_unselected():
    """不在 classes 列表中的类别像素 alpha 必须为 0"""
    from utils import deeplab_service

    mask = np.array([[0, 1], [2, 3]])
    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(
            Image.new("RGB", (2, 2)), MODEL_ID, WEIGHT_PATH, classes=[1, 3], weight_sha256=WEIGHT_SHA256
        )
        img = Image.open(io.BytesIO(png_bytes))
        pixels = list(img.getdata())
        assert pixels[0][3] == 0
        assert pixels[1][3] == 180
        assert pixels[2][3] == 0
        assert pixels[3][3] == 180


def test_segment_rgba_classes_none_renders_all_foreground():
    """classes=None 时与现有行为一致：所有非背景类 alpha=180"""
    from utils import deeplab_service

    mask = np.array([[0, 1], [2, 3]])
    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(
            Image.new("RGB", (2, 2)), MODEL_ID, WEIGHT_PATH, classes=None, weight_sha256=WEIGHT_SHA256
        )
        img = Image.open(io.BytesIO(png_bytes))
        pixels = list(img.getdata())
        assert pixels[0][3] == 0
        assert pixels[1][3] == 180
        assert pixels[2][3] == 180
        assert pixels[3][3] == 180


def test_segment_rgba_empty_classes_renders_nothing():
    """classes=[] 时所有像素全透明"""
    from utils import deeplab_service

    mask = np.array([[1, 2], [3, 4]])
    with patch.object(deeplab_service, "_load_model_unlocked", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(
            Image.new("RGB", (2, 2)), MODEL_ID, WEIGHT_PATH, classes=[], weight_sha256=WEIGHT_SHA256
        )
        img = Image.open(io.BytesIO(png_bytes))
        for pixel in img.getdata():
            assert pixel[3] == 0, f"空 classes 时所有像素应透明，得到 {pixel[3]}"


def test_model_cache_is_keyed_by_weight_content_not_database_id_or_path():
    from utils import deeplab_service

    first_model = _make_mock_model()
    second_model = _make_mock_model()
    deeplab_service.clear_model_cache()
    try:
        with patch.object(
            deeplab_service,
            "DeeplabV3",
            side_effect=[first_model, second_model],
        ) as constructor, patch.object(
            deeplab_service,
            "_predict_mask",
            return_value=np.zeros((1, 1), dtype=np.uint8),
        ):
            deeplab_service.segment_rgba_png(
                Image.new("RGB", (1, 1)), 1, "one.pth", weight_sha256="a" * 64
            )
            deeplab_service.segment_rgba_png(
                Image.new("RGB", (1, 1)), 2, "same-content.pt", weight_sha256="a" * 64
            )
            deeplab_service.segment_rgba_png(
                Image.new("RGB", (1, 1)), 3, "two.pt", weight_sha256="b" * 64
            )

        assert constructor.call_count == 2
        assert len(deeplab_service._model_cache) == 2
    finally:
        deeplab_service.clear_model_cache()


def test_model_cache_is_bounded_and_evicts_inactive_lru(monkeypatch):
    from utils import deeplab_service

    deeplab_service.clear_model_cache()
    monkeypatch.setattr(deeplab_service, "MODEL_CACHE_CAPACITY", 2)
    try:
        with patch.object(
            deeplab_service,
            "DeeplabV3",
            side_effect=[_make_mock_model(), _make_mock_model(), _make_mock_model()],
        ), patch.object(
            deeplab_service,
            "_predict_mask",
            return_value=np.zeros((1, 1), dtype=np.uint8),
        ):
            for index, digest in enumerate(("a", "b", "c"), start=1):
                deeplab_service.segment_rgba_png(
                    Image.new("RGB", (1, 1)),
                    index,
                    f"{index}.pth",
                    weight_sha256=digest * 64,
                )

        assert len(deeplab_service._model_cache) == 2
        assert all(key.weight_sha256 != "a" * 64 for key in deeplab_service._model_cache)
    finally:
        deeplab_service.clear_model_cache()


def test_background_model_reference_uses_inference_semaphore_per_tile(monkeypatch):
    from utils import deeplab_service

    class RecordingSemaphore:
        def __init__(self):
            self.entries = 0

        def __enter__(self):
            self.entries += 1

        def __exit__(self, exc_type, exc, traceback):
            return False

    semaphore = RecordingSemaphore()
    model = _make_mock_model()
    monkeypatch.setattr(deeplab_service, "_inference_semaphore", semaphore)
    deeplab_service.clear_model_cache()
    try:
        with patch.object(deeplab_service, "DeeplabV3", return_value=model), patch.object(
            deeplab_service.SlidingWindowPredictor,
            "_predict_tile",
            return_value=np.zeros((2, 2), dtype=np.uint8),
        ):
            with deeplab_service.model_tile_predictor(
                "one.pth", weight_sha256="a" * 64
            ) as predict_tile:
                predict_tile(np.zeros((2, 2, 3), dtype=np.uint8))
                predict_tile(np.zeros((2, 2, 3), dtype=np.uint8))

        assert semaphore.entries == 2
        assert deeplab_service._cache_lock is not deeplab_service._inference_semaphore
    finally:
        deeplab_service.clear_model_cache()


def test_bbox_inference_acquires_semaphore_per_tile(monkeypatch):
    from utils import deeplab_service

    class RecordingSemaphore:
        def __init__(self):
            self.entries = 0

        def __enter__(self):
            self.entries += 1

        def __exit__(self, exc_type, exc, traceback):
            return False

    class TwoTilePredictor:
        tile_size = 1
        stride = 1

        def __init__(self, model):
            pass

        def _predict_tile(self, tile):
            return np.zeros(tile.shape[:2], dtype=np.uint8)

    semaphore = RecordingSemaphore()
    monkeypatch.setattr(deeplab_service, "_inference_semaphore", semaphore)
    monkeypatch.setattr(deeplab_service, "SlidingWindowPredictor", TwoTilePredictor)
    deeplab_service.clear_model_cache()
    try:
        with patch.object(
            deeplab_service,
            "DeeplabV3",
            return_value=_make_mock_model(),
        ):
            deeplab_service.predict_mask(
                Image.new("RGB", (2, 1)),
                1,
                "one.pth",
                weight_sha256="a" * 64,
            )

        assert semaphore.entries == 2
    finally:
        deeplab_service.clear_model_cache()


def test_missing_weight_cannot_be_given_a_path_based_content_identity():
    from utils import deeplab_service

    with pytest.raises(FileNotFoundError):
        deeplab_service._file_sha256("missing-model.pth")


def test_render_mask_png_is_independent_from_inference():
    from utils import deeplab_service

    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    png_bytes = deeplab_service.render_mask_png(mask, _make_mock_model(), classes=[1, 3])
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    assert image.size == (2, 2)
    assert [pixel[3] for pixel in image.getdata()] == [0, 180, 0, 180]
