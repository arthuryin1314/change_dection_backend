import asyncio
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql

from crud.identification_results import SqlAlchemyClaimStore
from models.classification_results import ClassificationResult
from services.identification_results import Lease, ResultIdentity


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class RecordingSession:
    def __init__(self, result):
        self.result = result
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return ScalarResult(self.result)


def _identity():
    return ResultIdentity(
        user_id=7,
        image_content_sha256="a" * 64,
        weight_content_sha256="b" * 64,
        inference_parameters={"tile_size": 512, "overlap": 128},
    )


def test_result_identity_has_one_database_uniqueness_authority():
    constraints = [
        constraint
        for constraint in ClassificationResult.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    ]

    assert len(constraints) == 1
    assert constraints[0].name == "uq_classification_results_user_identity"
    assert [column.name for column in constraints[0].columns] == [
        "user_id",
        "identity_sha256",
    ]


def test_source_foreign_keys_do_not_delete_content_results():
    image_fk = next(iter(ClassificationResult.source_image_id.property.columns)).foreign_keys
    model_fk = next(iter(ClassificationResult.source_model_id.property.columns)).foreign_keys

    assert {foreign_key.ondelete for foreign_key in image_fk} == {"SET NULL"}
    assert {foreign_key.ondelete for foreign_key in model_fk} == {"SET NULL"}


def test_area_columns_and_migration_share_the_same_storage_contract():
    columns = ClassificationResult.__table__.columns
    assert columns.class_area_m2.type.__class__.__name__ == "JSON"
    assert columns.area_status.nullable is False
    assert columns.area_completed_at.nullable is True
    assert columns.area_failure_detail.nullable is True

    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in ClassificationResult.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "NOT_COMPUTED" in constraints["ck_classification_results_area_status"]
    assert "SUCCEEDED" in constraints["ck_classification_results_area_status"]
    assert "FAILED" in constraints["ck_classification_results_area_status"]

    migration = Path("migrations/002_classification_result_areas.sql").read_text(
        encoding="utf-8"
    )
    for column_name in (
        "class_area_m2",
        "area_status",
        "area_completed_at",
        "area_failure_detail",
    ):
        assert column_name in migration
    assert "conrelid = 'classification_results'::regclass" in migration


def test_insert_claim_uses_postgresql_conflict_handling_instead_of_check_then_insert():
    session = RecordingSession(None)
    store = SqlAlchemyClaimStore(session, source_image_id=11, source_model_id=12)
    now = datetime(2026, 9, 6, tzinfo=timezone.utc)

    result = asyncio.run(
        store.insert_processing(_identity(), Lease("worker", now, now))
    )
    sql = str(
        session.statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )

    assert result is None
    assert "ON CONFLICT (user_id, identity_sha256) DO NOTHING" in sql
    assert "RETURNING classification_results" in sql


def test_stale_takeover_is_one_conditional_update():
    session = RecordingSession(None)
    store = SqlAlchemyClaimStore(session, source_image_id=11, source_model_id=12)
    now = datetime(2026, 9, 6, tzinfo=timezone.utc)

    result = asyncio.run(
        store.take_over_if_reclaimable(
            7,
            _identity().sha256(),
            Lease("worker", now, now),
        )
    )
    sql = str(
        session.statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )

    assert result is None
    assert sql.startswith("UPDATE classification_results SET")
    assert "classification_results.status" in sql
    assert "classification_results.lease_expires_at" in sql
    assert "classification_results.lease_expires_at IS NULL" in sql
    assert "class_area_m2" in sql
    assert "area_status" in sql
    assert "area_completed_at" in sql
    assert "area_failure_detail" in sql
    assert "NOT_COMPUTED" in session.statements[0].compile(
        dialect=postgresql.dialect()
    ).params.values()
    assert "RETURNING classification_results" in sql
