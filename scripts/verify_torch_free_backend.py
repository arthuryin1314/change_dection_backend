"""Run startup and all tests against a disposable PostgreSQL database.

Run from the backend repository in its Torch-free environment after installing
requirements-dev.txt. DATABASE_URL (or .env) must identify a PostgreSQL account
allowed to create databases and enable PostGIS. The configured database is not
modified; only a uniquely named acceptance database is created and dropped.
"""

import asyncio
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from sqlalchemy.engine import make_url


def main() -> int:
    for package in ("torch", "torchvision"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(f"验收环境不能安装 {package}")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    configured = make_url(os.environ["DATABASE_URL"])
    admin_url = configured.set(drivername="postgresql", database="postgres")
    database_name = "torch_free_check_" + uuid4().hex
    admin = psycopg2.connect(admin_url.render_as_string(hide_password=False))
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
        print(f"已创建隔离验收库：{database_name}", flush=True)
        try:
            target = configured.set(database=database_name)
            os.environ["DATABASE_URL"] = target.render_as_string(hide_password=False)
            os.environ["RUN_POSTGRES_INTEGRATION"] = "1"
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["PYTHONUTF8"] = "1"
            database = psycopg2.connect(
                target.set(drivername="postgresql").render_as_string(hide_password=False)
            )
            try:
                database.autocommit = True
                with database.cursor() as cursor:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                # Startup cleanup jobs must also see isolated upload directories.
                with TemporaryDirectory(prefix="torch-free-check-") as scratch:
                    for variable, directory in (
                        ("CLASSIFICATION_RESULT_DIR", "classifications"),
                        ("CHANGE_INPUT_SNAPSHOT_DIR", "snapshots"),
                        ("CHANGE_RESULT_ARTIFACT_DIR", "artifacts"),
                    ):
                        os.environ[variable] = str(Path(scratch) / directory)
                    os.chdir(scratch)
                    try:
                        from fastapi.testclient import TestClient
                        from main import app
                        from config.db_config import async_engine

                        with TestClient(app) as client:
                            response = client.get("/openapi.json")
                            response.raise_for_status()
                        asyncio.run(async_engine.dispose(close=False))
                        print("无 Torch 的真实应用启动和关闭通过", flush=True)
                    finally:
                        os.chdir(root)
                    with database.cursor() as cursor:
                        for migration in sorted((root / "migrations").glob("*.sql")):
                            cursor.execute(migration.read_text(encoding="utf-8"))
                    result = subprocess.run(
                        [
                            sys.executable, "-m", "pytest", "tests", "-q",
                            "-p", "no:cacheprovider", "--tb=short",
                            "--basetemp=" + str(Path(scratch) / "pytest"),
                        ],
                        cwd=root,
                        env=os.environ,
                    )
                    return result.returncode
            finally:
                database.close()
        finally:
            with admin.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database_name))
                )
            print(f"已删除隔离验收库：{database_name}", flush=True)
    finally:
        admin.close()


if __name__ == "__main__":
    raise SystemExit(main())
