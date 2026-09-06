import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Protocol

from utils.classification_contract import (
    CLASSIFICATION_SCHEME_VERSION,
    GRID_POLICY_VERSION,
    PIPELINE_VERSION,
)


PROCESSING = "PROCESSING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"

LEASE_DURATION = timedelta(minutes=5)


@dataclass(frozen=True)
class ResultIdentity:
    user_id: int
    image_content_sha256: str
    weight_content_sha256: str
    inference_parameters: Mapping[str, object]
    classification_scheme_version: str = CLASSIFICATION_SCHEME_VERSION
    pipeline_version: str = PIPELINE_VERSION
    grid_policy_version: str = GRID_POLICY_VERSION

    def sha256(self) -> str:
        payload = {
            "classification_scheme_version": self.classification_scheme_version,
            "grid_policy_version": self.grid_policy_version,
            "image_content_sha256": self.image_content_sha256.lower(),
            "inference_parameters": self.inference_parameters,
            "pipeline_version": self.pipeline_version,
            "user_id": self.user_id,
            "weight_content_sha256": self.weight_content_sha256.lower(),
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Lease:
    owner: str
    now: datetime
    expires_at: datetime


@dataclass(frozen=True)
class IdentificationResultRecord:
    result_id: str
    user_id: int
    identity_sha256: str
    status: str
    lease_owner: str
    started_at: datetime
    heartbeat_at: datetime
    lease_expires_at: datetime | None


@dataclass(frozen=True)
class ResultClaim:
    record: IdentificationResultRecord
    should_start: bool


class ClaimStore(Protocol):
    async def insert_processing(
        self,
        identity: ResultIdentity,
        lease: Lease,
    ) -> IdentificationResultRecord | None: ...

    async def get_by_identity(
        self,
        user_id: int,
        identity_sha256: str,
    ) -> IdentificationResultRecord | None: ...

    async def take_over_if_reclaimable(
        self,
        user_id: int,
        identity_sha256: str,
        lease: Lease,
    ) -> IdentificationResultRecord | None: ...


async def claim_identification_result(
    store: ClaimStore,
    identity: ResultIdentity,
    lease_owner: str,
    now: datetime,
) -> ResultClaim:
    lease = Lease(lease_owner, now, now + LEASE_DURATION)
    inserted = await store.insert_processing(identity, lease)
    if inserted is not None:
        return ResultClaim(inserted, should_start=True)

    identity_sha256 = identity.sha256()
    existing = await store.get_by_identity(identity.user_id, identity_sha256)
    if existing is None:
        raise RuntimeError("唯一冲突后未能读取识别结果")
    if existing.status == SUCCEEDED:
        return ResultClaim(existing, should_start=False)
    if (
        existing.status == PROCESSING
        and existing.lease_expires_at is not None
        and existing.lease_expires_at > now
    ):
        return ResultClaim(existing, should_start=False)

    taken_over = await store.take_over_if_reclaimable(
        identity.user_id,
        identity_sha256,
        lease,
    )
    if taken_over is not None:
        return ResultClaim(taken_over, should_start=True)

    current = await store.get_by_identity(identity.user_id, identity_sha256)
    if current is None:
        raise RuntimeError("接管竞态后未能读取识别结果")
    return ResultClaim(current, should_start=False)
