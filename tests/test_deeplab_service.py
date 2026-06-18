import io
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


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

    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=np.zeros((4, 4), dtype=int)):
        result = deeplab_service.segment_rgba_png(Image.new("RGB", (4, 4)))
        assert isinstance(result, bytes)


def test_segment_rgba_output_mode_is_rgba():
    """Output PNG must be RGBA."""
    from utils import deeplab_service

    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=np.zeros((4, 4), dtype=int)):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (4, 4)))
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGBA"


def test_segment_rgba_background_alpha_is_zero():
    """Background class pixels must have alpha 0."""
    from utils import deeplab_service

    mask = np.array([[0, 0], [0, 0]])
    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)))
        img = Image.open(io.BytesIO(png_bytes))
        for pixel in img.getdata():
            assert pixel[3] == 0, f"background alpha should be 0, got {pixel[3]}"


def test_segment_rgba_foreground_alpha_is_180():
    """Foreground class pixels must have alpha 180."""
    from utils import deeplab_service

    mask = np.array([[1, 1], [1, 1]])
    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)))
        img = Image.open(io.BytesIO(png_bytes))
        for pixel in img.getdata():
            assert pixel[3] == 180, f"foreground alpha should be 180, got {pixel[3]}"
            assert pixel[:3] == (0, 0, 255), f"water color should be (0, 0, 255), got {pixel[:3]}"


def test_segment_rgba_mixed_classes():
    """Mixed background and foreground classes use the expected alpha values."""
    from utils import deeplab_service

    mask = np.array([[0, 1], [2, 3]])
    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)))
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
    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)), classes=[1, 3])
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
    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)), classes=None)
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
    with patch.object(deeplab_service, "get_model", return_value=_make_mock_model()), \
         patch.object(deeplab_service, "_predict_mask", return_value=mask):
        png_bytes = deeplab_service.segment_rgba_png(Image.new("RGB", (2, 2)), classes=[])
        img = Image.open(io.BytesIO(png_bytes))
        for pixel in img.getdata():
            assert pixel[3] == 0, f"空 classes 时所有像素应透明，得到 {pixel[3]}"
