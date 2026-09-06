from uuid import uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from models.classification_results import ClassificationResult
from services.identification_results import (
    FAILED,
    PROCESSING,
    IdentificationResultRecord,
    Lease,
    ResultIdentity,
)


def _record(row: ClassificationResult) -> IdentificationResultRecord:
    return IdentificationResultRecord(
        result_id=row.id,
        user_id=row.user_id,
        identity_sha256=row.identity_sha256,
        status=row.status,
        lease_owner=row.lease_owner,
        started_at=row.started_at,
        heartbeat_at=row.heartbeat_at,
        lease_expires_at=row.lease_expires_at,
    )


class SqlAlchemyClaimStore:
    def __init__(
        self,
        session: AsyncSession,
        *,
        source_image_id: int,
        source_model_id: int,
    ):
        self.session = session
        self.source_image_id = source_image_id
        self.source_model_id = source_model_id

    async def insert_processing(
        self,
        identity: ResultIdentity,
        lease: Lease,
    ) -> IdentificationResultRecord | None:
        statement = (
            insert(ClassificationResult)
            .values(
                id=uuid4().hex,
                user_id=identity.user_id,
                source_image_id=self.source_image_id,
                source_model_id=self.source_model_id,
                identity_sha256=identity.sha256(),
                image_content_sha256=identity.image_content_sha256.lower(),
                weight_content_sha256=identity.weight_content_sha256.lower(),
                inference_parameters=dict(identity.inference_parameters),
                classification_scheme_version=identity.classification_scheme_version,
                pipeline_version=identity.pipeline_version,
                grid_policy_version=identity.grid_policy_version,
                status=PROCESSING,
                started_at=lease.now,
                heartbeat_at=lease.now,
                lease_expires_at=lease.expires_at,
                lease_owner=lease.owner,
            )
            .on_conflict_do_nothing(
                index_elements=["user_id", "identity_sha256"],
            )
            .returning(ClassificationResult)
        )
        result = await self.session.execute(statement)
        row = result.scalar_one_or_none()
        return _record(row) if row is not None else None

    async def get_by_identity(
        self,
        user_id: int,
        identity_sha256: str,
    ) -> IdentificationResultRecord | None:
        statement = select(ClassificationResult).where(
            ClassificationResult.user_id == user_id,
            ClassificationResult.identity_sha256 == identity_sha256,
        )
        result = await self.session.execute(statement)
        row = result.scalar_one_or_none()
        return _record(row) if row is not None else None

    async def take_over_if_reclaimable(
        self,
        user_id: int,
        identity_sha256: str,
        lease: Lease,
    ) -> IdentificationResultRecord | None:
        reclaimable = or_(
            ClassificationResult.status == FAILED,
            and_(
                ClassificationResult.status == PROCESSING,
                or_(
                    ClassificationResult.lease_expires_at.is_(None),
                    ClassificationResult.lease_expires_at <= lease.now,
                ),
            ),
        )
        statement = (
            update(ClassificationResult)
            .where(
                ClassificationResult.user_id == user_id,
                ClassificationResult.identity_sha256 == identity_sha256,
                reclaimable,
            )
            .values(
                status=PROCESSING,
                lease_owner=lease.owner,
                started_at=lease.now,
                heartbeat_at=lease.now,
                lease_expires_at=lease.expires_at,
                completed_at=None,
                failure_detail=None,
            )
            .returning(ClassificationResult)
        )
        result = await self.session.execute(statement)
        row = result.scalar_one_or_none()
        return _record(row) if row is not None else None
