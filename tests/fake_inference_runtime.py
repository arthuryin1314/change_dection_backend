"""Small runtime test double; model packages and real weights are never loaded."""

from contextlib import contextmanager

import numpy as np

from services.inference_runtime import UnavailableInferenceRuntime


@contextmanager
def constant_predictor(weight_file_path, *, weight_sha256):
    yield lambda rgb: np.full(rgb.shape[:2], 5, dtype=np.uint8)


class FakeInferenceRuntime(UnavailableInferenceRuntime):
    def __init__(self, predictor_factory=constant_predictor):
        self.predictor_factory = predictor_factory

    def ensure_available(self) -> None:
        pass

    def open_predictor(self, weight_file_path, *, weight_sha256):
        return self.predictor_factory(weight_file_path, weight_sha256=weight_sha256)
