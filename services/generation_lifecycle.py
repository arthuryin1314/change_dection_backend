from datetime import datetime, timezone

from sqlalchemy import or_, update

from models.classification_results import ClassificationResult
from services.classification_generation import GenerationOutcome
from services.identification_results import FAILED, LEASE_DURATION, PROCESSING, SUCCEEDED


class LeaseLostError(RuntimeError):
    pass


async def recover_expired_processing_results(session_factory, *, now: datetime) -> int:
    statement = (
        update(ClassificationResult)
        .where(
            ClassificationResult.status == PROCESSING,
            or_(
                ClassificationResult.lease_expires_at.is_(None),
                ClassificationResult.lease_expires_at <= now,
            ),
        )
        .values(
            status=FAILED,
            completed_at=now,
            failure_detail="处理进程中断，可重新提交",
            lease_expires_at=None,
        )
    )
    async with session_factory() as session:
        result = await session.execute(statement)
        await session.commit()
        return result.rowcount


class SqlAlchemyGenerationLifecycle:
    def __init__(
        self,
        session_factory,
        result_id: str,
        lease_owner: str,
        *,
        clock=lambda: datetime.now(timezone.utc),
    ):
        self.session_factory = session_factory
        self.result_id = result_id
        self.lease_owner = lease_owner
        self.clock = clock

    def _owned_processing_update(self):
        return update(ClassificationResult).where(
            ClassificationResult.id == self.result_id,
            ClassificationResult.status == PROCESSING,
            ClassificationResult.lease_owner == self.lease_owner,
        )

    async def _execute_owned_update(self, statement) -> None:
        async with self.session_factory() as session:
            result = await session.execute(statement)
            if result.rowcount != 1:
                raise LeaseLostError("识别结果租约已被其他任务接管")
            await session.commit()

    async def heartbeat(self) -> None:
        now = self.clock()
        await self._execute_owned_update(
            self._owned_processing_update().values(
                heartbeat_at=now,
                lease_expires_at=now + LEASE_DURATION,
            )
        )

    async def mark_succeeded(self, outcome: GenerationOutcome) -> None:
        now = self.clock()
        metadata = outcome.metadata
        metrics = outcome.metrics
        await self._execute_owned_update(
            self._owned_processing_update().values(
                status=SUCCEEDED,
                completed_at=now,
                heartbeat_at=now,
                lease_expires_at=None,
                failure_detail=None,
                classes_path=str(outcome.stored.classes_path),
                valid_mask_path=str(outcome.stored.valid_mask_path),
                crs=metadata.crs,
                transform=list(metadata.transform),
                raster_width=metadata.width,
                raster_height=metadata.height,
                resolution=list(metadata.resolution),
                bounds=list(metadata.bounds),
                generation_metrics={
                    "source_read_seconds": metrics.source_read_seconds,
                    "inference_seconds": metrics.inference_seconds,
                    "compressed_write_seconds": metrics.compressed_write_seconds,
                    "model_load_seconds": metrics.model_load_seconds,
                    "total_seconds": metrics.total_seconds,
                    "peak_rss_bytes": metrics.peak_rss_bytes,
                    "peak_gpu_bytes": metrics.peak_gpu_bytes,
                    "total_tiles": metrics.total_tiles,
                    "effective_tiles": metrics.effective_tiles,
                    "skipped_tiles": metrics.skipped_tiles,
                },
            )
        )

    async def mark_failed(self, detail: str) -> None:
        now = self.clock()
        await self._execute_owned_update(
            self._owned_processing_update().values(
                status=FAILED,
                completed_at=now,
                heartbeat_at=now,
                lease_expires_at=None,
                failure_detail=detail[:4000],
            )
        )
