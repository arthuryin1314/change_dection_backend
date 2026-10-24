import asyncio
from io import BytesIO

from fastapi import UploadFile
import pytest

from utils.model_asset_storage import ModelAssetTooLargeError, save_upload_file


def test_streamed_size_limit_cleans_partial_asset(tmp_path):
    upload = UploadFile(file=BytesIO(b"12345"), filename="weights.pth", size=None)
    with pytest.raises(ModelAssetTooLargeError):
        asyncio.run(save_upload_file(upload, tmp_path / "assets", 7, 4))
    assert list((tmp_path / "assets" / "7").iterdir()) == []
