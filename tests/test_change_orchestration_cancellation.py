import asyncio
import os

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

from services.change_orchestration import _await_artifact_task


def test_artifact_thread_task_finishes_before_cancellation_propagates():
    asyncio.run(_cancelled_outer())


async def _cancelled_outer():
    finished = asyncio.Event()

    async def writer():
        await asyncio.sleep(0.03)
        finished.set()
        return {"artifact_version": "change-artifacts-v1"}

    outer = asyncio.create_task(_await_artifact_task(asyncio.create_task(writer())))
    await asyncio.sleep(0)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert finished.is_set()
