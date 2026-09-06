import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from services.identification_results import (
    FAILED,
    PROCESSING,
    SUCCEEDED,
    IdentificationResultRecord,
    ResultIdentity,
    claim_identification_result,
)


NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)
IDENTITY = ResultIdentity(
    user_id=7,
    image_content_sha256="a" * 64,
    weight_content_sha256="b" * 64,
    inference_parameters={"overlap": 128, "tile_size": 512},
)


class InMemoryClaimStore:
    def __init__(self, existing=None):
        self._record = existing
        self._lock = asyncio.Lock()
        self._next_id = 1

    async def insert_processing(self, identity, lease):
        async with self._lock:
            if self._record is not None:
                return None
            self._record = IdentificationResultRecord(
                result_id=str(self._next_id),
                user_id=identity.user_id,
                identity_sha256=identity.sha256(),
                status=PROCESSING,
                lease_owner=lease.owner,
                started_at=lease.now,
                heartbeat_at=lease.now,
                lease_expires_at=lease.expires_at,
            )
            self._next_id += 1
            return self._record

    async def get_by_identity(self, user_id, identity_sha256):
        async with self._lock:
            if (
                self._record is not None
                and self._record.user_id == user_id
                and self._record.identity_sha256 == identity_sha256
            ):
                return self._record
            return None

    async def take_over_if_reclaimable(self, user_id, identity_sha256, lease):
        async with self._lock:
            record = self._record
            if record.status == FAILED or (
                record.status == PROCESSING
                and (
                    record.lease_expires_at is None
                    or record.lease_expires_at <= lease.now
                )
            ):
                self._record = replace(
                    record,
                    status=PROCESSING,
                    lease_owner=lease.owner,
                    started_at=lease.now,
                    heartbeat_at=lease.now,
                    lease_expires_at=lease.expires_at,
                )
                return self._record
            return None


def _existing_record(status, *, lease_expires_at=None):
    return IdentificationResultRecord(
        result_id="existing-result",
        user_id=IDENTITY.user_id,
        identity_sha256=IDENTITY.sha256(),
        status=status,
        lease_owner="old-worker",
        started_at=NOW - timedelta(minutes=5),
        heartbeat_at=NOW - timedelta(minutes=5),
        lease_expires_at=lease_expires_at,
    )


def test_identity_uses_content_and_normalized_inference_not_source_ids():
    reordered = ResultIdentity(
        user_id=7,
        image_content_sha256="a" * 64,
        weight_content_sha256="b" * 64,
        inference_parameters={"tile_size": 512, "overlap": 128},
    )
    changed_image = replace(IDENTITY, image_content_sha256="c" * 64)
    changed_weight = replace(IDENTITY, weight_content_sha256="d" * 64)

    assert IDENTITY.sha256() == reordered.sha256()
    assert IDENTITY.sha256() != changed_image.sha256()
    assert IDENTITY.sha256() != changed_weight.sha256()


def test_duplicate_succeeded_result_is_reused_without_starting_work():
    store = InMemoryClaimStore(_existing_record(SUCCEEDED))

    claim = asyncio.run(
        claim_identification_result(store, IDENTITY, "new-worker", NOW)
    )

    assert claim.record.result_id == "existing-result"
    assert claim.record.status == SUCCEEDED
    assert claim.should_start is False


def test_concurrent_duplicate_claims_create_one_result_and_one_worker():
    store = InMemoryClaimStore()

    async def claim_twice():
        return await asyncio.gather(
            claim_identification_result(store, IDENTITY, "worker-a", NOW),
            claim_identification_result(store, IDENTITY, "worker-b", NOW),
        )

    first, second = asyncio.run(claim_twice())

    assert first.record.result_id == second.record.result_id
    assert sum(claim.should_start for claim in (first, second)) == 1


def test_unexpired_processing_lease_is_not_taken_over():
    store = InMemoryClaimStore(
        _existing_record(PROCESSING, lease_expires_at=NOW + timedelta(seconds=1))
    )

    claim = asyncio.run(
        claim_identification_result(store, IDENTITY, "new-worker", NOW)
    )

    assert claim.record.lease_owner == "old-worker"
    assert claim.should_start is False


def test_expired_processing_lease_is_atomically_taken_over():
    store = InMemoryClaimStore(
        _existing_record(PROCESSING, lease_expires_at=NOW - timedelta(seconds=1))
    )

    claim = asyncio.run(
        claim_identification_result(store, IDENTITY, "new-worker", NOW)
    )

    assert claim.record.result_id == "existing-result"
    assert claim.record.lease_owner == "new-worker"
    assert claim.record.started_at == NOW
    assert claim.should_start is True


def test_processing_result_with_missing_lease_is_atomically_taken_over():
    store = InMemoryClaimStore(
        _existing_record(PROCESSING, lease_expires_at=None)
    )

    claim = asyncio.run(
        claim_identification_result(store, IDENTITY, "new-worker", NOW)
    )

    assert claim.record.lease_owner == "new-worker"
    assert claim.should_start is True


def test_failed_result_keeps_its_id_when_retried():
    store = InMemoryClaimStore(_existing_record(FAILED))

    claim = asyncio.run(
        claim_identification_result(store, IDENTITY, "retry-worker", NOW)
    )

    assert claim.record.result_id == "existing-result"
    assert claim.record.status == PROCESSING
    assert claim.record.lease_owner == "retry-worker"
    assert claim.should_start is True
