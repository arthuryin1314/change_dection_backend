import argparse
import ctypes
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


CLASS_COUNT = 6


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


def memory_bytes() -> tuple[int, int]:
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
        raise OSError("cannot read process memory counters")
    return counters.WorkingSetSize, counters.PeakWorkingSetSize


def count_window(
    counts: np.ndarray,
    classes: np.ndarray,
    valid: np.ndarray,
    window: Window,
    axis: str,
) -> int:
    selected = valid == 1
    selected_classes = classes[selected]
    if selected_classes.size and int(selected_classes.max()) >= CLASS_COUNT:
        raise ValueError("classification contains a class outside 0-5")

    height, width = classes.shape
    if axis == "column":
        positions = np.broadcast_to(np.arange(width), (height, width))[selected]
        encoded = selected_classes.astype(np.int64) * width + positions
        local = np.bincount(encoded, minlength=CLASS_COUNT * width).reshape(
            CLASS_COUNT, width
        )
        col_off = int(window.col_off)
        counts[:, col_off : col_off + width] += local
    else:
        positions = np.broadcast_to(np.arange(height)[:, np.newaxis], (height, width))[
            selected
        ]
        encoded = selected_classes.astype(np.int64) * height + positions
        local = np.bincount(encoded, minlength=CLASS_COUNT * height).reshape(
            CLASS_COUNT, height
        )
        row_off = int(window.row_off)
        counts[:, row_off : row_off + height] += local
    return int(selected_classes.size)


def scan(args) -> None:
    started = perf_counter()
    initial_rss, _ = memory_bytes()
    with rasterio.Env(GDAL_CACHEMAX=args.gdal_cache_mb * 1024 * 1024), rasterio.open(
        args.classes
    ) as classes_ds, rasterio.open(args.mask) as mask_ds:
        if (
            classes_ds.width != mask_ds.width
            or classes_ds.height != mask_ds.height
            or classes_ds.transform != mask_ds.transform
            or classes_ds.crs != mask_ds.crs
        ):
            raise ValueError("classification and valid mask grids differ")
        axis = "column" if classes_ds.crs.is_projected else "row"
        axis_length = classes_ds.width if axis == "column" else classes_ds.height
        counts = np.zeros((CLASS_COUNT, axis_length), dtype=np.int64)

        if args.layout == "blocks":
            windows = (window for _, window in classes_ds.block_windows(1))
        else:
            windows = (
                Window(0, row, classes_ds.width, min(args.strip_height, classes_ds.height - row))
                for row in range(0, classes_ds.height, args.strip_height)
            )

        window_count = 0
        valid_pixels = 0
        for window in windows:
            classes = classes_ds.read(1, window=window)
            valid = mask_ds.read(1, window=window)
            valid_pixels += count_window(counts, classes, valid, window, axis)
            window_count += 1

    current_rss, peak_rss = memory_bytes()
    print(
        json.dumps(
            {
                "kind": "scan",
                "layout": args.layout,
                "strip_height": args.strip_height if args.layout == "strips" else None,
                "gdal_cache_mb": args.gdal_cache_mb,
                "shape": [classes_ds.height, classes_ds.width],
                "axis": axis,
                "windows": window_count,
                "valid_pixels": valid_pixels,
                "class_pixels": counts.sum(axis=1).tolist(),
                "seconds": perf_counter() - started,
                "initial_rss_bytes": initial_rss,
                "current_rss_bytes": current_rss,
                "peak_rss_bytes": peak_rss,
                "peak_delta_bytes": max(0, peak_rss - initial_rss),
            }
        )
    )


def hash_file(path: Path) -> tuple[str, float]:
    started = perf_counter()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest(), perf_counter() - started


def hash_benchmark(args) -> None:
    first_digest, first_seconds = hash_file(args.path)
    second_digest, second_seconds = hash_file(args.path)
    stat = args.path.stat()
    started = perf_counter()
    cached = (
        first_digest
        if stat.st_size == args.path.stat().st_size
        and stat.st_mtime_ns == args.path.stat().st_mtime_ns
        else None
    )
    cached_seconds = perf_counter() - started
    print(
        json.dumps(
            {
                "kind": "hash",
                "path": str(args.path),
                "size_bytes": stat.st_size,
                "first_full_seconds": first_seconds,
                "second_full_seconds": second_seconds,
                "metadata_cache_seconds": cached_seconds,
                "digests_match": first_digest == second_digest == cached,
            }
        )
    )


def handle_probe(_args) -> None:
    root = Path(tempfile.mkdtemp(prefix="issue8-handle-probe-"))
    published = root / "result"
    quarantine = root / ".result.corrupt"
    published.mkdir()
    profile = {
        "driver": "GTiff",
        "width": 256,
        "height": 256,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4528",
        "transform": Affine(0.8, 0, 500000, 0, -0.8, 3200000),
    }
    for name in ("classes.tif", "valid_mask.tif"):
        with rasterio.open(published / name, "w", **profile) as dataset:
            dataset.write(np.ones((1, 256, 256), dtype=np.uint8))

    replace_result = "not attempted"
    remove_result = "not attempted"
    classes_ds = rasterio.open(published / "classes.tif")
    mask_ds = rasterio.open(published / "valid_mask.tif")
    try:
        try:
            published.replace(quarantine)
            replace_result = "succeeded"
        except OSError as exc:
            replace_result = f"failed: {type(exc).__name__}: {exc}"
        target = quarantine if quarantine.exists() else published
        try:
            shutil.rmtree(target)
            remove_result = "succeeded"
        except OSError as exc:
            remove_result = f"failed: {type(exc).__name__}: {exc}"
    finally:
        classes_ds.close()
        mask_ds.close()
        shutil.rmtree(root, ignore_errors=True)
    print(
        json.dumps(
            {
                "kind": "handle_probe",
                "replace_while_open": replace_result,
                "rmtree_while_open": remove_result,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--classes", type=Path, required=True)
    scan_parser.add_argument("--mask", type=Path, required=True)
    scan_parser.add_argument("--layout", choices=("blocks", "strips"), required=True)
    scan_parser.add_argument("--strip-height", type=int, default=256)
    scan_parser.add_argument("--gdal-cache-mb", type=int, default=128)
    scan_parser.set_defaults(run=scan)

    hash_parser = subparsers.add_parser("hash")
    hash_parser.add_argument("--path", type=Path, required=True)
    hash_parser.set_defaults(run=hash_benchmark)

    handle_parser = subparsers.add_parser("handle-probe")
    handle_parser.set_defaults(run=handle_probe)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
