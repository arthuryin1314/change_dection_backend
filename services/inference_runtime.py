"""Boundary between the Web backend and model execution.

The production implementation is deliberately unavailable until the inference
Worker is connected. Tests inject a runtime without loading model frameworks.
"""

from contextlib import AbstractContextManager
from typing import Callable, Protocol

import numpy as np
from PIL import Image


class InferenceUnavailableError(RuntimeError):
    error_code = "INFERENCE_UNAVAILABLE"
    message = "推理服务暂不可用"

    def __init__(self):
        super().__init__(self.message)


class ModelLoadError(RuntimeError):
    """The selected model cannot be loaded by the runtime."""


class InferenceRuntime(Protocol):
    def ensure_available(self) -> None: ...

    def open_predictor(
        self, weight_file_path: str, *, weight_sha256: str
    ) -> AbstractContextManager[Callable[[np.ndarray], np.ndarray]]: ...

    def segment_rgba_png(
        self,
        image: Image.Image,
        model_id: int,
        weight_file_path: str,
        classes: list[int] | None,
        *,
        weight_sha256: str,
    ) -> bytes: ...


class UnavailableInferenceRuntime:
    def ensure_available(self) -> None:
        raise InferenceUnavailableError()

    def open_predictor(
        self, weight_file_path: str, *, weight_sha256: str
    ) -> AbstractContextManager[Callable[[np.ndarray], np.ndarray]]:
        raise InferenceUnavailableError()

    def segment_rgba_png(
        self,
        image: Image.Image,
        model_id: int,
        weight_file_path: str,
        classes: list[int] | None,
        *,
        weight_sha256: str,
    ) -> bytes:
        raise InferenceUnavailableError()


UNAVAILABLE_RUNTIME = UnavailableInferenceRuntime()


def get_inference_runtime() -> InferenceRuntime:
    return UNAVAILABLE_RUNTIME
