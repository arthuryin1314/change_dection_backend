import asyncio
import ctypes
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Callable, Protocol
from uuid import uuid4

import rasterio
import torch

from utils.classification_storage import (
    AtomicClassificationWriter,
    RasterGrid,
    StoredClassification,
    validate_stored_classification,
)
from utils.classification_result_lock import classification_result_lock
from utils.deeplab_service import model_tile_predictor
from utils.streaming_classification import (
    ClassificationGenerationCancelled,
    StreamingMetrics,
    stream_classification_bands,
)


HEARTBEAT_INTERVAL_SECONDS = 60
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationRequest:
    result_id: str
    image_path: str | Path
    weight_file_path: str
    weight_sha256: str
    storage_root: str | Path


@dataclass(frozen=True)
class SpatialMetadata:
    width: int
    height: int
    crs: str
    transform: tuple[float, float, float, float, float, float]
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class GenerationOutcome:
    stored: StoredClassification
    metadata: SpatialMetadata
    metrics: StreamingMetrics


class GenerationLifecycle(Protocol):
    async def heartbeat(self) -> None: ...

    async def mark_succeeded(self, outcome: GenerationOutcome) -> None: ...

    async def mark_failed(self, detail: str) -> None: ...


def _raise_if_cancelled(cancellation_event: Event | None) -> None:
    if cancellation_event is not None and cancellation_event.is_set():
        raise ClassificationGenerationCancelled("识别结果生成已取消")


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        succeeded = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            raise OSError("无法读取进程峰值内存")
        return counters.PeakWorkingSetSize

    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def _metadata(dataset) -> SpatialMetadata:
    transform = dataset.transform
    return SpatialMetadata(
        width=dataset.width,
        height=dataset.height,
        crs=dataset.crs.to_string(),
        transform=(
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ),
        resolution=(dataset.res[0], dataset.res[1]),
        bounds=(
            dataset.bounds.left,
            dataset.bounds.bottom,
            dataset.bounds.right,
            dataset.bounds.top,
        ),
    )


def _generate_classification_files_unlocked(
    request: GenerationRequest,
    *,
    progress_callback: Callable[[StreamingMetrics], None] | None = None,
    cancellation_event: Event | None = None,
    publication_guard: Callable[[], None] | None = None,
) -> GenerationOutcome:
    _raise_if_cancelled(cancellation_event)
    total_started = perf_counter()
    metrics = StreamingMetrics()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with rasterio.open(request.image_path) as source:
        if source.crs is None:
            raise ValueError("源影像缺少坐标系")
        if source.count < 3:
            raise ValueError("源影像必须至少包含三个波段")
        if any(dtype != "uint8" for dtype in source.dtypes[:3]):
            raise ValueError("源影像前三个波段必须为 uint8")

        grid = RasterGrid(
            width=source.width,
            height=source.height,
            crs=source.crs,
            transform=source.transform,
        )
        metadata = _metadata(source)
        final_directory = Path(request.storage_root) / request.result_id
        existing = StoredClassification(
            directory=final_directory,
            classes_path=final_directory / "classes.tif",
            valid_mask_path=final_directory / "valid_mask.tif",
        )
        if validate_stored_classification(existing, grid, verify_pixels=False):
            metrics.total_seconds = perf_counter() - total_started
            metrics.peak_rss_bytes = _peak_rss_bytes()
            if torch.cuda.is_available():
                metrics.peak_gpu_bytes = torch.cuda.max_memory_allocated()
            return GenerationOutcome(existing, metadata, metrics)

        quarantine = None
        if final_directory.exists():
            quarantine = final_directory.with_name(
                f".{request.result_id}.{uuid4().hex}.corrupt"
            )
            final_directory.replace(quarantine)
        try:
            model_started = perf_counter()
            with model_tile_predictor(
                request.weight_file_path,
                weight_sha256=request.weight_sha256,
            ) as predict_tile:
                _raise_if_cancelled(cancellation_event)
                metrics.model_load_seconds = perf_counter() - model_started
                writer = AtomicClassificationWriter(
                    request.storage_root,
                    request.result_id,
                    grid,
                )
                open_started = perf_counter()
                with writer:
                    metrics.compressed_write_seconds += perf_counter() - open_started
                    for band in stream_classification_bands(
                        source,
                        predict_tile,
                        metrics=metrics,
                        cancellation_event=cancellation_event,
                    ):
                        _raise_if_cancelled(cancellation_event)
                        write_started = perf_counter()
                        writer.write(band.window, band.classes, band.valid_mask)
                        metrics.compressed_write_seconds += (
                            perf_counter() - write_started
                        )
                        if progress_callback is not None:
                            progress_callback(metrics)
                    _raise_if_cancelled(cancellation_event)
                    finalize_started = perf_counter()

                    def before_publish() -> None:
                        _raise_if_cancelled(cancellation_event)
                        if publication_guard is not None:
                            publication_guard()
                        _raise_if_cancelled(cancellation_event)

                    stored = writer.finalize(
                        before_publish=before_publish
                    )
                    metrics.compressed_write_seconds += perf_counter() - finalize_started
        except BaseException:
            if quarantine is not None and quarantine.exists():
                if final_directory.exists():
                    _remove_quarantine_best_effort(quarantine)
                else:
                    try:
                        quarantine.replace(final_directory)
                    except OSError:
                        logger.warning(
                            "旧识别结果隔离目录恢复失败: %s",
                            quarantine,
                            exc_info=True,
                        )
            raise
        else:
            if quarantine is not None and quarantine.exists():
                _remove_quarantine_best_effort(quarantine)

    metrics.total_seconds = perf_counter() - total_started
    metrics.peak_rss_bytes = _peak_rss_bytes()
    if torch.cuda.is_available():
        metrics.peak_gpu_bytes = torch.cuda.max_memory_allocated()
    return GenerationOutcome(stored, metadata, metrics)


def _remove_quarantine_best_effort(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        logger.warning(
            "旧识别结果隔离目录清理失败: %s",
            path,
            exc_info=True,
        )


def generate_classification_files(
    request: GenerationRequest,
    *,
    progress_callback: Callable[[StreamingMetrics], None] | None = None,
    cancellation_event: Event | None = None,
    publication_guard: Callable[[], None] | None = None,
) -> GenerationOutcome:
    result_directory = Path(request.storage_root) / request.result_id
    with classification_result_lock(result_directory):
        return _generate_classification_files_unlocked(
            request,
            progress_callback=progress_callback,
            cancellation_event=cancellation_event,
            publication_guard=publication_guard,
        )


async def _heartbeat_loop(lifecycle: GenerationLifecycle) -> None:
    while True:
        await lifecycle.heartbeat()
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)


async def run_generation(
    request: GenerationRequest,
    lifecycle: GenerationLifecycle,
) -> GenerationOutcome:
    cancellation_event = Event()
    event_loop = asyncio.get_running_loop()

    def publication_guard() -> None:
        renewed = asyncio.run_coroutine_threadsafe(
            lifecycle.heartbeat(),
            event_loop,
        )
        renewed.result()

    heartbeat_task = asyncio.create_task(_heartbeat_loop(lifecycle))
    worker_task = asyncio.create_task(
        asyncio.to_thread(
            generate_classification_files,
            request,
            cancellation_event=cancellation_event,
            publication_guard=publication_guard,
        )
    )
    try:
        done, _ = await asyncio.wait(
            (worker_task, heartbeat_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            heartbeat_error = heartbeat_task.exception()
            if heartbeat_error is not None:
                cancellation_event.set()
                try:
                    await worker_task
                except Exception:
                    pass
                raise heartbeat_error
        outcome = await worker_task
        await lifecycle.mark_succeeded(outcome)
        return outcome
    except asyncio.CancelledError:
        cancellation_event.set()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        await lifecycle.mark_failed("服务关闭，任务可重新提交")
        raise
    except Exception as exc:
        await lifecycle.mark_failed(str(exc))
        raise
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
