from dataclasses import dataclass
from threading import Event
from time import perf_counter
from typing import Callable, Iterator

import numpy as np
from rasterio.windows import Window

from utils.classification_source import read_rgb_and_valid_mask
from utils.classification_contract import INFERENCE_OVERLAP, INFERENCE_TILE_SIZE


class ClassificationGenerationCancelled(RuntimeError):
    pass


def _raise_if_cancelled(cancellation_event: Event | None) -> None:
    if cancellation_event is not None and cancellation_event.is_set():
        raise ClassificationGenerationCancelled("识别结果生成已取消")


@dataclass
class StreamingMetrics:
    source_read_seconds: float = 0.0
    inference_seconds: float = 0.0
    compressed_write_seconds: float = 0.0
    model_load_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_rss_bytes: int = 0
    peak_gpu_bytes: int = 0
    total_tiles: int = 0
    effective_tiles: int = 0
    skipped_tiles: int = 0


@dataclass(frozen=True)
class ClassificationBand:
    window: Window
    classes: np.ndarray
    valid_mask: np.ndarray


def _axis_windows(length: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    windows = []
    for offset in range(0, length, stride):
        end = min(offset + tile_size, length)
        start = max(0, end - tile_size)
        windows.append((start, end))
    return windows


def stream_classification_bands(
    dataset,
    predict_tile: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int = INFERENCE_TILE_SIZE,
    overlap: int = INFERENCE_OVERLAP,
    metrics: StreamingMetrics | None = None,
    cancellation_event: Event | None = None,
) -> Iterator[ClassificationBand]:
    if tile_size <= overlap:
        raise ValueError("tile_size 必须大于 overlap")
    measured = metrics if metrics is not None else StreamingMetrics()
    stride = tile_size - overlap
    y_windows = _axis_windows(dataset.height, tile_size, stride)
    x_windows = _axis_windows(dataset.width, tile_size, stride)

    buffer_start = 0
    classes_buffer = np.zeros((0, dataset.width), dtype=np.uint8)
    valid_buffer = np.zeros((0, dataset.width), dtype=np.uint8)

    for y_index, (y1, y2) in enumerate(y_windows):
        _raise_if_cancelled(cancellation_event)
        buffer_end = buffer_start + classes_buffer.shape[0]
        if y2 > buffer_end:
            extra_rows = y2 - buffer_end
            classes_buffer = np.concatenate(
                (
                    classes_buffer,
                    np.zeros((extra_rows, dataset.width), dtype=np.uint8),
                ),
                axis=0,
            )
            valid_buffer = np.concatenate(
                (
                    valid_buffer,
                    np.zeros((extra_rows, dataset.width), dtype=np.uint8),
                ),
                axis=0,
            )

        relative_y1 = y1 - buffer_start
        relative_y2 = y2 - buffer_start
        for x1, x2 in x_windows:
            _raise_if_cancelled(cancellation_event)
            measured.total_tiles += 1
            read_started = perf_counter()
            source = read_rgb_and_valid_mask(
                dataset,
                Window(x1, y1, x2 - x1, y2 - y1),
            )
            measured.source_read_seconds += perf_counter() - read_started

            if not np.any(source.valid_mask):
                measured.skipped_tiles += 1
                classes = np.zeros(source.valid_mask.shape, dtype=np.uint8)
            else:
                measured.effective_tiles += 1
                inference_started = perf_counter()
                classes = predict_tile(source.rgb)
                measured.inference_seconds += perf_counter() - inference_started
                _raise_if_cancelled(cancellation_event)
                if classes.shape != source.valid_mask.shape:
                    raise ValueError("模型输出形状与输入 tile 不一致")
                if np.any(classes < 0) or np.any(classes > 5):
                    raise ValueError("模型输出类别必须在 0-5 之间")
                classes = np.where(source.valid_mask == 1, classes, 0).astype(
                    np.uint8,
                    copy=False,
                )

            classes_buffer[relative_y1:relative_y2, x1:x2] = classes
            valid_buffer[relative_y1:relative_y2, x1:x2] = source.valid_mask

        if y_index + 1 < len(y_windows):
            flush_end = min(start for start, _ in y_windows[y_index + 1 :])
        else:
            flush_end = dataset.height
        flush_rows = flush_end - buffer_start
        if flush_rows > 0:
            _raise_if_cancelled(cancellation_event)
            yield ClassificationBand(
                window=Window(0, buffer_start, dataset.width, flush_rows),
                classes=classes_buffer[:flush_rows].copy(),
                valid_mask=valid_buffer[:flush_rows].copy(),
            )
            classes_buffer = classes_buffer[flush_rows:]
            valid_buffer = valid_buffer[flush_rows:]
            buffer_start = flush_end
