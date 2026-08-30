import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
os.environ.setdefault("GEOSERVER_URL", "http://example.com/geoserver")
os.environ.setdefault("GEOSERVER_USER", "admin")
os.environ.setdefault("GEOSERVER_PASSWORD", "geoserver")
os.environ.setdefault("GEOSERVER_WORKSPACE", "ws")

from router.users import account_closure


def _db(*, commit=None):
    return SimpleNamespace(
        commit=AsyncMock(side_effect=commit),
        rollback=AsyncMock(),
    )


def _stubs(monkeypatch, *, user=True, cleanup=False):
    monkeypatch.setattr(
        account_closure,
        "get_user_by_id",
        AsyncMock(return_value=SimpleNamespace(id=7) if user else None),
    )
    monkeypatch.setattr(
        account_closure,
        "delete_user_record",
        AsyncMock(return_value={"user_id": 7}),
    )
    monkeypatch.setattr(
        account_closure.image_lifecycle,
        "prepare_user_image_deletion",
        AsyncMock(return_value=SimpleNamespace(operation_id="op-1", deleted_count=3)),
        raising=False,
    )
    finish_cleanup = AsyncMock(return_value=cleanup)
    monkeypatch.setattr(
        account_closure.image_lifecycle,
        "finish_cleanup",
        finish_cleanup,
        raising=False,
    )
    return finish_cleanup


def test_closing_missing_account_rolls_back_without_committing(monkeypatch):
    finish_cleanup = _stubs(monkeypatch, user=False)
    db = _db()

    with pytest.raises(HTTPException) as error:
        asyncio.run(account_closure.close_account(db, 7))

    assert error.value.status_code == 404
    db.rollback.assert_awaited_once()
    db.commit.assert_not_awaited()
    finish_cleanup.assert_not_awaited()


def test_closing_account_does_not_cleanup_when_commit_fails(monkeypatch):
    finish_cleanup = _stubs(monkeypatch)
    cancel_cleanup = Mock()
    monkeypatch.setattr(account_closure.image_lifecycle, "cancel_cleanup", cancel_cleanup)
    db = _db(commit=RuntimeError("commit failed"))

    with pytest.raises(HTTPException) as error:
        asyncio.run(account_closure.close_account(db, 7))

    assert error.value.status_code == 500
    db.commit.assert_awaited_once()
    db.rollback.assert_awaited_once()
    finish_cleanup.assert_not_awaited()
    cancel_cleanup.assert_called_once_with("op-1")


def test_closing_account_succeeds_when_post_commit_cleanup_fails(monkeypatch):
    finish_cleanup = _stubs(monkeypatch, cleanup=False)
    db = _db()

    result = asyncio.run(account_closure.close_account(db, 7))

    assert result == {"id": 7, "deleted_images": 3}
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()
    finish_cleanup.assert_awaited_once_with("op-1")
