import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from services.generation_lifecycle import (
    SqlAlchemyGenerationLifecycle,
    recover_expired_processing_results,
)


class ExecuteResult:
    rowcount = 1


class RecordingSession:
    def __init__(self):
        self.statements = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return ExecuteResult()

    async def commit(self):
        self.commits += 1


class SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback):
        return False


def _factory(session):
    return lambda: SessionContext(session)


def _sql(statement):
    return str(statement.compile(dialect=postgresql.dialect()))


def test_heartbeat_only_renews_the_owned_processing_lease():
    session = RecordingSession()
    lifecycle = SqlAlchemyGenerationLifecycle(
        _factory(session),
        "result-1",
        "worker-1",
        clock=lambda: datetime(2026, 9, 6, tzinfo=timezone.utc),
    )

    asyncio.run(lifecycle.heartbeat())

    sql = _sql(session.statements[0])
    assert "classification_results.id" in sql
    assert "classification_results.lease_owner" in sql
    assert "classification_results.status" in sql
    assert session.commits == 1


def test_success_transition_writes_spatial_contract_in_same_update():
    session = RecordingSession()
    lifecycle = SqlAlchemyGenerationLifecycle(
        _factory(session),
        "result-1",
        "worker-1",
        clock=lambda: datetime(2026, 9, 6, tzinfo=timezone.utc),
    )
    outcome = SimpleNamespace(
        stored=SimpleNamespace(
            classes_path=Path("results/result-1/classes.tif"),
            valid_mask_path=Path("results/result-1/valid_mask.tif"),
        ),
        metadata=SimpleNamespace(
            width=6,
            height=5,
            crs="EPSG:4528",
            transform=(0.8, 0.0, 500000.0, 0.0, -0.8, 3200000.0),
            resolution=(0.8, 0.8),
            bounds=(500000.0, 3199996.0, 500004.8, 3200000.0),
        ),
        metrics=SimpleNamespace(
            source_read_seconds=1.0,
            inference_seconds=2.0,
            compressed_write_seconds=3.0,
            model_load_seconds=0.5,
            total_seconds=6.5,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            total_tiles=1,
            effective_tiles=1,
            skipped_tiles=0,
        ),
    )

    asyncio.run(lifecycle.mark_succeeded(outcome))

    statement = session.statements[0]
    sql = _sql(statement)
    params = statement.compile(dialect=postgresql.dialect()).params
    assert "classes_path" in sql
    assert "valid_mask_path" in sql
    assert "generation_metrics" in sql
    assert "SUCCEEDED" in params.values()
    set_clause = sql.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
    assert "lease_owner" not in set_clause
    assert session.commits == 1


def test_failure_transition_does_not_publish_paths():
    session = RecordingSession()
    lifecycle = SqlAlchemyGenerationLifecycle(
        _factory(session),
        "result-1",
        "worker-1",
        clock=lambda: datetime(2026, 9, 6, tzinfo=timezone.utc),
    )

    asyncio.run(lifecycle.mark_failed("inference failed"))

    sql = _sql(session.statements[0])
    assert "failure_detail" in sql
    assert "classes_path" not in sql
    assert "valid_mask_path" not in sql
    assert session.commits == 1


def test_startup_recovery_marks_expired_processing_rows_retryable():
    session = RecordingSession()
    now = datetime(2026, 9, 6, tzinfo=timezone.utc)

    recovered = asyncio.run(
        recover_expired_processing_results(_factory(session), now=now)
    )

    sql = _sql(session.statements[0])
    params = session.statements[0].compile(dialect=postgresql.dialect()).params
    assert recovered == 1
    assert "classification_results.lease_expires_at" in sql
    assert "classification_results.lease_expires_at IS NULL" in sql
    assert "FAILED" in params.values()
    assert session.commits == 1
