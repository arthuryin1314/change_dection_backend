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
