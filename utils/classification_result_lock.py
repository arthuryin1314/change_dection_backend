import os
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep


LOCK_RETRY_SECONDS = 0.05
LOCK_TIMEOUT_SECONDS = 30


def _try_lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def classification_result_lock(
    result_directory: str | Path,
    *,
    timeout_seconds: float = LOCK_TIMEOUT_SECONDS,
):
    directory = Path(result_directory)
    lock_directory = directory.parent / ".locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{directory.name}.lock"
    with lock_path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()

        deadline = monotonic() + timeout_seconds
        while True:
            try:
                _try_lock(handle)
                break
            except OSError as exc:
                if monotonic() >= deadline:
                    raise TimeoutError("等待识别结果文件锁超时") from exc
                sleep(LOCK_RETRY_SECONDS)
        try:
            yield
        finally:
            _unlock(handle)
