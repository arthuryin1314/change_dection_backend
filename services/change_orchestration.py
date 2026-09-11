import asyncio
import hashlib
import json
import logging
import os
import shutil
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from affine import Affine
from config.db_config import AsyncSessionLocal
from crud import change_results as change_crud
from crud import classification_results as classification_crud
from crud.identification_results import SqlAlchemyClaimStore
from services.change_input_freeze import FrozenFile, freeze_file
from services.change_inputs import ResolvedChangeInputs, ResolvedPeriod
from services.classification_generation import (
    GenerationRequest,
    InputVersionChangedError,
    run_generation,
)
from services.generation_lifecycle import SqlAlchemyGenerationLifecycle
from services.identification_results import (
    FAILED,
    PROCESSING,
    SUCCEEDED,
    ResultIdentity,
    claim_identification_result,
)
from utils.change_result_errors import (
    CHANGE_RESULT_FAILED,
    CHANGE_RESULT_INTERRUPTED,
    IDENTIFICATION_EXECUTION_FAILED,
    IDENTIFICATION_RESULT_BUSY,
    INPUT_VERSION_UNAVAILABLE,
)
from utils.classification_contract import (
    CLASSIFICATION_SCHEME_VERSION,
    GRID_POLICY_VERSION as CLASSIFICATION_GRID_POLICY_VERSION,
    PIPELINE_VERSION,
)
from utils.classification_result_lock import classification_result_lock
from utils.classification_storage import (
    RasterGrid,
    StoredClassification,
    validate_stored_classification,
)
from utils.transition_matrix import (
    CALCULATION_VERSION,
    GRID_POLICY_VERSION,
    TransitionMatrixError,
    compute_transition_matrix_m2,
)
from utils.result_source import ResultSourceSnapshot, result_source_snapshot


logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL_SECONDS = 15
STALE_SCAN_INTERVAL_SECONDS = 30
POLL_MIN_SECONDS = 2
POLL_MAX_SECONDS = 5
IDENTIFICATION_WAIT_SECONDS = float(
    os.environ.get("CHANGE_IDENTIFICATION_WAIT_SECONDS", "120")
)
DB_POLL_LIMIT = 5
PERIODS = ("before", "after")
SNAPSHOT_ROOT = Path(
    os.environ.get("CHANGE_INPUT_SNAPSHOT_DIR", "uploads/change_input_snapshots")
)
_tasks: dict[int, asyncio.Task] = {}
_stale_scan_task: asyncio.Task | None = None
_poll_slots: asyncio.Semaphore | None = None


class InputVersionError(RuntimeError):
    pass


class IdentificationExecutionError(RuntimeError):
    def __init__(self, period: str, detail: str):
        super().__init__(detail)
        self.period = period


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _poll_semaphore() -> asyncio.Semaphore:
    global _poll_slots
    if _poll_slots is None:
        _poll_slots = asyncio.Semaphore(DB_POLL_LIMIT)
    return _poll_slots


def _frozen_payload(value: FrozenFile) -> dict:
    stat = value.path.stat()
    return {
        "path": str(value.path),
        "sha256": value.sha256,
        "size": value.size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _freeze(result_id: int, submitted: dict, periods: tuple[str, ...]) -> dict:
    if not periods:
        return {}
    directory = SNAPSHOT_ROOT / str(result_id)
    weight_source = Path(submitted["model"]["weight_path"])
    frozen = {}
    try:
        for period in periods:
            source = Path(submitted[period]["path"])
            frozen[period] = _frozen_payload(
                freeze_file(
                    source,
                    directory,
                    f"{period}{source.suffix.lower()}",
                )
            )
        weight = freeze_file(
            weight_source,
            directory,
            f"weight{weight_source.suffix.lower()}",
        )
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    frozen["model"] = _frozen_payload(weight)
    return frozen


def _cleanup(result_id: int) -> None:
    root = SNAPSHOT_ROOT.resolve()
    target = (SNAPSHOT_ROOT / str(result_id)).resolve()
    if target.parent != root:
        raise RuntimeError("变化任务冻结目录越出存储根")
    shutil.rmtree(target, ignore_errors=True)


def _identity(
    user_id: int,
    image_sha256: str,
    weight_sha256: str,
    contract: dict,
) -> ResultIdentity:
    return ResultIdentity(
        user_id=user_id,
        image_content_sha256=image_sha256,
        weight_content_sha256=weight_sha256,
        inference_parameters=contract["inference_parameters"],
        classification_scheme_version=contract["classification_scheme_version"],
        pipeline_version=contract["pipeline_version"],
        grid_policy_version=contract["grid_policy_version"],
    )


def _orchestration_identity(
    before_identity_sha256: str,
    after_identity_sha256: str,
    contract: dict,
) -> str:
    payload = {
        "before_identity_sha256": before_identity_sha256,
        "after_identity_sha256": after_identity_sha256,
        "calculation_version": contract["calculation_version"],
        "grid_policy_version": contract["grid_policy_version"],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_snapshot(submitted: dict, period: str) -> ResultSourceSnapshot:
    return result_source_snapshot(
        submitted[f"{period}_image_id"],
        submitted[period].get("name"),
        submitted["model_id"],
        submitted["model"].get("name"),
    )


def _resolved_period(row, submitted: dict, period: str) -> ResolvedPeriod:
    return ResolvedPeriod.from_row(row, _source_snapshot(submitted, period))


def _validate_identity_contract(submitted: dict) -> dict:
    identity_contract = submitted["identity_contract"]
    parameters = identity_contract["inference_parameters"]
    if (
        identity_contract["classification_scheme_version"]
        != CLASSIFICATION_SCHEME_VERSION
        or identity_contract["pipeline_version"] != PIPELINE_VERSION
        or identity_contract["grid_policy_version"]
        != CLASSIFICATION_GRID_POLICY_VERSION
        or set(parameters) != {"tile_size", "overlap"}
        or not isinstance(parameters["tile_size"], int)
        or not isinstance(parameters["overlap"], int)
        or parameters["overlap"] < 0
        or parameters["tile_size"] <= parameters["overlap"]
    ):
        raise InputVersionError("提交时的识别执行契约已不可用")
    return identity_contract


def _validate_analysis_contract(submitted: dict) -> dict:
    analysis_contract = submitted["analysis_contract"]
    if analysis_contract != {
        "calculation_version": CALCULATION_VERSION,
        "grid_policy_version": GRID_POLICY_VERSION,
    }:
        raise InputVersionError("提交时的变化分析执行契约已不可用")
    return analysis_contract


def _stored_result_is_reusable(row) -> bool:
    if (
        row.classes_path is None
        or row.valid_mask_path is None
        or row.crs is None
        or row.transform is None
        or row.raster_width is None
        or row.raster_height is None
    ):
        return False
    stored = StoredClassification(
        directory=Path(row.classes_path).parent,
        classes_path=Path(row.classes_path),
        valid_mask_path=Path(row.valid_mask_path),
    )
    grid = RasterGrid(
        width=row.raster_width,
        height=row.raster_height,
        crs=row.crs,
        transform=Affine(*row.transform),
    )
    with classification_result_lock(stored.directory, timeout_seconds=2):
        return validate_stored_classification(stored, grid, verify_pixels=False)


async def _load_classification(result_id: str, user_id: int):
    async with _poll_semaphore():
        async with AsyncSessionLocal() as db:
            return await classification_crud.get_result_by_id(db, result_id, user_id)


async def _load_reusable_classification(result_id: str | None, user_id: int):
    if result_id is None:
        return None
    current = await _load_classification(result_id, user_id)
    if current is None or current.status != SUCCEEDED:
        return None
    reusable = await asyncio.to_thread(_stored_result_is_reusable, current)
    return current if reusable else None


async def _ensure_classification(
    *,
    change_result_id: int,
    user_id: int,
    source_image_id: int,
    source_model_id: int,
    source_snapshot: ResultSourceSnapshot,
    image: dict,
    weight: dict,
    identity: ResultIdentity,
    period: str,
):
    phase_name = f"ENSURING_{period.upper()}"
    async with AsyncSessionLocal() as db:
        row = await change_crud.get_by_id(db, change_result_id)
        if row is None:
            raise RuntimeError("变化任务不存在")
        if not await change_crud.set_phase(
            db,
            result_id=change_result_id,
            user_id=user_id,
            owner=row.lease_owner,
            phase=phase_name,
            now=_now(),
        ):
            raise RuntimeError("变化任务租约已失效")
        store = SqlAlchemyClaimStore(
            db,
            source_image_id=source_image_id,
            source_model_id=source_model_id,
            source_snapshot=source_snapshot,
        )
        claim = await claim_identification_result(
            store,
            identity,
            row.lease_owner,
            _now(),
        )
        await db.commit()

    if claim.record.status == SUCCEEDED:
        current = await _load_classification(claim.record.result_id, user_id)
        try:
            reusable = await asyncio.to_thread(_stored_result_is_reusable, current)
        except TimeoutError:
            reusable = False
        if reusable:
            return current
        async with AsyncSessionLocal() as db:
            await classification_crud.invalidate_succeeded_result(
                db,
                claim.record.result_id,
                user_id,
            )
            await db.commit()
        return await _ensure_classification(
            change_result_id=change_result_id,
            user_id=user_id,
            source_image_id=source_image_id,
            source_model_id=source_model_id,
            source_snapshot=source_snapshot,
            image=image,
            weight=weight,
            identity=identity,
            period=period,
        )

    if claim.should_start:
        request = GenerationRequest(
            result_id=claim.record.result_id,
            image_path=image["path"],
            image_sha256=image["sha256"],
            weight_file_path=weight["path"],
            weight_sha256=weight["sha256"],
            inference_parameters=identity.inference_parameters,
            storage_root=os.environ.get(
                "CLASSIFICATION_RESULT_DIR",
                "uploads/classification_results",
            ),
        )
        lifecycle = SqlAlchemyGenerationLifecycle(
            AsyncSessionLocal,
            request.result_id,
            claim.record.lease_owner,
        )
        try:
            await run_generation(request, lifecycle)
        except InputVersionChangedError as exc:
            raise InputVersionError(str(exc)) from exc
        except Exception as exc:
            raise IdentificationExecutionError(period, str(exc)) from exc

    delay = POLL_MIN_SECONDS
    loop = asyncio.get_running_loop()
    deadline = loop.time() + IDENTIFICATION_WAIT_SECONDS

    async def wait_for_retry() -> None:
        nonlocal delay
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("等待识别结果完成超时")
        await asyncio.sleep(min(delay, remaining))
        delay = min(POLL_MAX_SECONDS, delay + 1)

    while True:
        current = await _load_classification(claim.record.result_id, user_id)
        if current is None:
            raise IdentificationExecutionError(period, "识别结果记录不存在")
        if current.status == FAILED:
            raise IdentificationExecutionError(
                period,
                current.failure_detail or "识别执行失败",
            )
        if current.status == SUCCEEDED:
            try:
                reusable = await asyncio.to_thread(_stored_result_is_reusable, current)
            except TimeoutError:
                await wait_for_retry()
                continue
            if reusable:
                return current
            raise IdentificationExecutionError(period, "识别结果文件缺失或损坏")
        await wait_for_retry()


async def _heartbeat(result_id: int, user_id: int, owner: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        async with AsyncSessionLocal() as db:
            row = await change_crud.get_by_id(db, result_id)
            if row is None or row.status != PROCESSING or row.lease_owner != owner:
                return
            active = await change_crud.heartbeat(
                row.request_id,
                user_id,
                owner,
                _now(),
                db,
            )
            if not active:
                await db.rollback()
                return
            await db.commit()


async def _mark_failure(
    result_id: int,
    user_id: int,
    owner: str,
    *,
    error_code: str,
    message: str,
    error_data: dict,
    http_status: int,
) -> None:
    async with AsyncSessionLocal() as db:
        row = await change_crud.get_by_id(db, result_id)
        if row is None:
            return
        await change_crud.mark_failed(
            db,
            request_id=row.request_id,
            user_id=user_id,
            owner=owner,
            http_status=http_status,
            error_code=error_code,
            message=message,
            error_data=error_data,
            now=_now(),
        )


async def _execute(result_id: int, user_id: int, owner: str) -> None:
    async with AsyncSessionLocal() as db:
        row = await change_crud.get_by_id(db, result_id)
        if row is None or row.status != PROCESSING or row.lease_owner != owner:
            return
        submitted = dict(row.submitted_inputs)
        carried_result_ids = {
            "before": row.before_result_id,
            "after": row.after_result_id,
        }

    reused = {
        period: await _load_reusable_classification(
            carried_result_ids[period],
            user_id,
        )
        for period in PERIODS
    }
    missing_periods = tuple(period for period in PERIODS if reused[period] is None)
    analysis_contract = _validate_analysis_contract(submitted)
    identity_contract = (
        _validate_identity_contract(submitted)
        if missing_periods
        else submitted["identity_contract"]
    )
    frozen = await asyncio.to_thread(
        _freeze,
        result_id,
        submitted,
        missing_periods,
    )
    generated_identities = {
        period: _identity(
            user_id,
            frozen[period]["sha256"],
            frozen["model"]["sha256"],
            identity_contract,
        )
        for period in missing_periods
    }
    identity_sha256 = {
        period: (
            reused[period].identity_sha256
            if reused[period] is not None
            else generated_identities[period].sha256()
        )
        for period in PERIODS
    }
    full_identity = _orchestration_identity(
        identity_sha256["before"],
        identity_sha256["after"],
        analysis_contract,
    )
    async with AsyncSessionLocal() as db:
        winner = await change_crud.set_frozen_identity(
            db,
            result_id=result_id,
            user_id=user_id,
            owner=owner,
            orchestration_identity_sha256=full_identity,
            before_identity_sha256=identity_sha256["before"],
            after_identity_sha256=identity_sha256["after"],
            frozen_inputs=frozen,
            now=_now(),
        )
    if winner is None or winner.id != result_id:
        return

    period_results = {}
    for period in PERIODS:
        period_result = reused[period]
        if period_result is None:
            period_result = await _ensure_classification(
                change_result_id=result_id,
                user_id=user_id,
                source_image_id=submitted[f"{period}_image_id"],
                source_model_id=submitted["model_id"],
                source_snapshot=_source_snapshot(submitted, period),
                image=frozen[period],
                weight=frozen["model"],
                identity=generated_identities[period],
                period=period,
            )
        period_results[period] = period_result
        async with AsyncSessionLocal() as db:
            if not await change_crud.mark_period_ready(
                db,
                result_id=result_id,
                user_id=user_id,
                owner=owner,
                period=period,
                result=period_result,
                now=_now(),
            ):
                return

    async with AsyncSessionLocal() as db:
        if not await change_crud.set_phase(
            db,
            result_id=result_id,
            user_id=user_id,
            owner=owner,
            phase="COMPUTING_MATRIX",
            now=_now(),
        ):
            return

    resolved = ResolvedChangeInputs(
        before=_resolved_period(period_results["before"], submitted, "before"),
        after=_resolved_period(period_results["after"], submitted, "after"),
    )
    result = await asyncio.to_thread(
        compute_transition_matrix_m2,
        resolved.before.classes_path,
        resolved.before.valid_mask_path,
        resolved.after.classes_path,
        resolved.after.valid_mask_path,
    )
    async with AsyncSessionLocal() as db:
        row = await change_crud.get_by_id(db, result_id)
        if row is None:
            return
        await change_crud.mark_succeeded(
            db,
            request_id=row.request_id,
            user_id=user_id,
            owner=owner,
            resolved=resolved,
            result=result,
            now=_now(),
        )


async def run_change_orchestration(result_id: int, user_id: int, owner: str) -> None:
    heartbeat = asyncio.create_task(_heartbeat(result_id, user_id, owner))
    try:
        await _execute(result_id, user_id, owner)
    except asyncio.CancelledError:
        await _mark_failure(
            result_id,
            user_id,
            owner,
            error_code=CHANGE_RESULT_INTERRUPTED,
            message="服务关闭，变化分析可重新提交",
            error_data={"retryable": True},
            http_status=409,
        )
        raise
    except IdentificationExecutionError as exc:
        await _mark_failure(
            result_id,
            user_id,
            owner,
            error_code=IDENTIFICATION_EXECUTION_FAILED,
            message="识别执行失败",
            error_data={
                "periods": {
                    exc.period: {
                        "reason": str(exc),
                        "status": FAILED,
                    }
                },
                "retryable": True,
            },
            http_status=422,
        )
    except (FileNotFoundError, InputVersionError) as exc:
        await _mark_failure(
            result_id,
            user_id,
            owner,
            error_code=INPUT_VERSION_UNAVAILABLE,
            message="任务提交时的输入版本已不可用",
            error_data={"reason": str(exc), "retryable": False},
            http_status=409,
        )
    except TimeoutError:
        await _mark_failure(
            result_id,
            user_id,
            owner,
            error_code=IDENTIFICATION_RESULT_BUSY,
            message="识别结果正在使用，请稍后重试",
            error_data={"retryable": True},
            http_status=409,
        )
    except TransitionMatrixError as exc:
        await _mark_failure(
            result_id,
            user_id,
            owner,
            error_code=exc.error_code,
            message=str(exc),
            error_data={"retryable": exc.error_code == IDENTIFICATION_RESULT_BUSY},
            http_status=409 if exc.error_code == IDENTIFICATION_RESULT_BUSY else 422,
        )
    except Exception:
        logger.exception("变化分析后台编排失败: result_id=%s", result_id)
        await _mark_failure(
            result_id,
            user_id,
            owner,
            error_code=CHANGE_RESULT_FAILED,
            message="变化分析执行失败",
            error_data={"retryable": True},
            http_status=500,
        )
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        await asyncio.to_thread(_cleanup, result_id)


def _task_done(result_id: int, task: asyncio.Task) -> None:
    _tasks.pop(result_id, None)
    if task.cancelled():
        return
    exception = task.exception()
    if exception is not None:
        logger.error(
            "变化分析任务退出异常: result_id=%s",
            result_id,
            exc_info=(type(exception), exception, exception.__traceback__),
        )


def schedule_change_orchestration(result_id: int, user_id: int, owner: str) -> None:
    if result_id in _tasks:
        return
    task = asyncio.create_task(run_change_orchestration(result_id, user_id, owner))
    _tasks[result_id] = task
    task.add_done_callback(lambda done: _task_done(result_id, done))


async def _stale_scan_loop() -> None:
    while True:
        await asyncio.sleep(STALE_SCAN_INTERVAL_SECONDS)
        async with AsyncSessionLocal() as db:
            await change_crud.expire_stale(db, now=_now())
            await db.commit()


def start_stale_scan() -> None:
    global _stale_scan_task
    if _stale_scan_task is None:
        _stale_scan_task = asyncio.create_task(_stale_scan_loop())


async def stop_change_orchestration() -> None:
    global _stale_scan_task
    if _stale_scan_task is not None:
        _stale_scan_task.cancel()
        with suppress(asyncio.CancelledError):
            await _stale_scan_task
        _stale_scan_task = None
    tasks = list(_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
