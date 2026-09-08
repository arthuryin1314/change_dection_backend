import asyncio
from datetime import datetime, timezone

from sqlalchemy.dialects import postgresql

from crud.classification_results import (
    invalidate_succeeded_result,
    mark_area_failed,
    mark_area_succeeded,
)


class ExecuteResult:
    def __init__(self, rowcount=1):
        self.rowcount = rowcount


class RecordingSession:
    def __init__(self, rowcount=1):
        self.statements = []
        self.rowcount = rowcount

    async def execute(self, statement):
        self.statements.append(statement)
        return ExecuteResult(self.rowcount)


def test_invalidating_classification_clears_derived_area():
    session = RecordingSession()

    changed = asyncio.run(invalidate_succeeded_result(session, "result-1", 7))

    statement = session.statements[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    params = statement.compile(dialect=postgresql.dialect()).params
    assert changed is True
    assert "class_area_m2" in sql
    assert "area_status" in sql
    assert "area_completed_at" in sql
    assert "area_failure_detail" in sql
    assert "NOT_COMPUTED" in params.values()


def _generation_token():
    return datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


def _assert_generation_guard(statement):
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "classification_results.id" in sql
    assert "classification_results.user_id" in sql
    assert "classification_results.status" in sql
    assert "classification_results.lease_owner" in sql
    assert "classification_results.completed_at" in sql


def test_area_success_is_written_only_to_the_generation_that_was_scanned():
    session = RecordingSession()

    changed = asyncio.run(
        mark_area_succeeded(
            session,
            result_id="result-1",
            user_id=7,
            lease_owner="worker-1",
            completed_at=_generation_token(),
            class_area_m2=[1, 2, 3, 4, 5, 6],
            area_completed_at=_generation_token(),
        )
    )

    statement = session.statements[0]
    params = statement.compile(dialect=postgresql.dialect()).params
    assert changed is True
    _assert_generation_guard(statement)
    assert "SUCCEEDED" in params.values()
    assert [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] in params.values()


def test_area_failure_is_written_only_to_the_generation_that_was_scanned():
    session = RecordingSession()

    changed = asyncio.run(
        mark_area_failed(
            session,
            result_id="result-1",
            user_id=7,
            lease_owner="worker-1",
            completed_at=_generation_token(),
            detail="unsupported grid",
            area_completed_at=_generation_token(),
        )
    )

    statement = session.statements[0]
    params = statement.compile(dialect=postgresql.dialect()).params
    assert changed is True
    _assert_generation_guard(statement)
    assert "FAILED" in params.values()
    assert "unsupported grid" in params.values()


def test_old_area_worker_cannot_write_after_classification_restarts():
    session = RecordingSession(rowcount=0)

    changed = asyncio.run(
        mark_area_succeeded(
            session,
            result_id="result-1",
            user_id=7,
            lease_owner="old-worker",
            completed_at=_generation_token(),
            class_area_m2=[0, 0, 0, 0, 0, 0],
            area_completed_at=_generation_token(),
        )
    )

    assert changed is False
