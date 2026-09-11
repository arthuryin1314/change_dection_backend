import os
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest
from dotenv import load_dotenv
from psycopg2 import sql


load_dotenv()

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_INTEGRATION") != "1",
    reason="需要显式启用隔离 PostgreSQL 集成测试",
)


def _database_url() -> str:
    return os.environ["DATABASE_URL"].replace(
        "postgresql+asyncpg://",
        "postgresql://",
        1,
    )


def test_history_migration_adds_snapshot_and_both_idempotent_sort_indexes():
    schema_name = f"history_migration_{uuid4().hex}"
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
            )
            cursor.execute(
                sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
            )
            cursor.execute(
                """
                CREATE TABLE classification_results (
                    id VARCHAR(32) PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    status VARCHAR(16) NOT NULL,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE change_results (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    status VARCHAR(16) NOT NULL,
                    completed_at TIMESTAMPTZ
                )
                """
            )
            migration = Path("migrations/006_result_history.sql").read_text(
                encoding="utf-8"
            )
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                """
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'classification_results'
                  AND column_name = 'source_snapshot'
                """,
                (schema_name,),
            )
            assert cursor.fetchone() == ("YES",)
            cursor.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = %s
                  AND indexname IN (
                    'ix_classification_results_user_history',
                    'ix_change_results_user_history'
                  )
                ORDER BY indexname
                """,
                (schema_name,),
            )
            indexes = dict(cursor.fetchall())
            assert set(indexes) == {
                "ix_change_results_user_history",
                "ix_classification_results_user_history",
            }
            for definition in indexes.values():
                assert "user_id" in definition
                assert "completed_at DESC" in definition
                assert "id DESC" in definition
                assert "WHERE" in definition
                assert "status" in definition
                assert "SUCCEEDED" in definition
    finally:
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SET search_path TO public")
            cursor.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )
        connection.commit()
        connection.close()


def test_history_name_backfill_recovers_legacy_classification_and_change_names():
    schema_name = f"history_name_backfill_{uuid4().hex}"
    connection = psycopg2.connect(_database_url())
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
            )
            cursor.execute(
                sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
            )
            cursor.execute(
                """
                CREATE TABLE images (
                    id INTEGER PRIMARY KEY,
                    image_name VARCHAR(255) NOT NULL
                );
                CREATE TABLE model_library (
                    id INTEGER PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL
                );
                CREATE TABLE classification_results (
                    id VARCHAR(32) PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    source_image_id INTEGER,
                    source_model_id INTEGER,
                    source_snapshot JSON,
                    status VARCHAR(16) NOT NULL,
                    completed_at TIMESTAMPTZ
                );
                CREATE TABLE change_results (
                    id BIGINT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    before_image_id INTEGER,
                    after_image_id INTEGER,
                    source_model_id INTEGER,
                    before_result_id VARCHAR(32),
                    after_result_id VARCHAR(32),
                    before_snapshot JSON,
                    after_snapshot JSON,
                    status VARCHAR(16) NOT NULL,
                    completed_at TIMESTAMPTZ
                );
                INSERT INTO images (id, image_name)
                VALUES (1, '前期影像'), (2, '后期影像');
                INSERT INTO model_library (id, model_name)
                VALUES (11, '地物分类模型');
                INSERT INTO classification_results (
                    id, user_id, source_image_id, source_model_id,
                    source_snapshot, status, completed_at
                ) VALUES
                    ('before-result', 7, 1, 11, NULL, 'SUCCEEDED', NOW()),
                    ('after-result', 7, 2, 11, 'null', 'SUCCEEDED', NOW());
                INSERT INTO change_results (
                    id, user_id, before_image_id, after_image_id, source_model_id,
                    before_result_id, after_result_id,
                    before_snapshot, after_snapshot, status, completed_at
                ) VALUES (
                    41,
                    7,
                    1,
                    2,
                    11,
                    'before-result',
                    'after-result',
                    '{"source": null, "source_image_id": 1, "source_model_id": 11}',
                    '{"source": null, "source_image_id": 2, "source_model_id": 11}',
                    'SUCCEEDED',
                    NOW()
                );
                """
            )
            migration = Path(
                "migrations/007_backfill_result_history_names.sql"
            ).read_text(encoding="utf-8")
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                """
                SELECT
                    source_snapshot->'image'->>'name',
                    source_snapshot->'model'->>'name'
                FROM classification_results
                ORDER BY id
                """
            )
            assert cursor.fetchall() == [
                ("后期影像", "地物分类模型"),
                ("前期影像", "地物分类模型"),
            ]
            cursor.execute(
                """
                SELECT
                    before_snapshot->'source'->'image'->>'name',
                    before_snapshot->'source'->'model'->>'name',
                    after_snapshot->'source'->'image'->>'name',
                    after_snapshot->'source'->'model'->>'name'
                FROM change_results
                WHERE id = 41
                """
            )
            assert cursor.fetchone() == (
                "前期影像",
                "地物分类模型",
                "后期影像",
                "地物分类模型",
            )
    finally:
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SET search_path TO public")
            cursor.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )
        connection.commit()
        connection.close()
