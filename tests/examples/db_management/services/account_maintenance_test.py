from __future__ import annotations

import asyncio
import unittest
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from pydantic import ValidationError

from examples.db_management.services import account_sync_steps
from examples.db_management.services.account_maintenance import (
    AccountMaintenance,
)
from examples.db_management.services.account_retention import (
    AccountMaintenanceSettings,
)
from examples.db_management.services.account_sync_steps import (
    AccountSynchronizer,
)
from examples.db_management.services.account_sync_steps import AccountSyncPlan


async def _owner_sessions(ids, /):
    yield SimpleNamespace(scalars=AsyncMock(return_value=ids))


async def _cleanup_timeouts_cannot_block_account_cleanup():
    await asyncio.Event().wait()


async def _cleanup_timeouts_cannot_block_account_process(
    synchronized,
    /,
    _,
):
    synchronized.set()


async def _slow_account_does_not_block_process(fast, /, identifier):
    if identifier == 'slow':
        await asyncio.Event().wait()
    else:
        fast.set()


async def _shutdown_drains_active_sync_and_process(
    cancelled,
    entered,
    /,
    _,
):
    entered.set()
    try:
        await asyncio.Event().wait()
    finally:
        cancelled.set()


async def _sync_budget_includes_selecting_pending_blocked_sessions():
    await asyncio.Event().wait()
    yield None


class AccountMaintenanceTest(unittest.IsolatedAsyncioTestCase):
    def owner(self, ids):
        sessions = asynccontextmanager(partial(_owner_sessions, ids))

        return SimpleNamespace(
            settings=AccountMaintenanceSettings(
                sync_interval_seconds=0.005,
                cleanup_interval_seconds=0.005,
                sync_budget_seconds=0.1,
                cleanup_budget_seconds=0.01,
            ),
            sessions=sessions,
            stop=asyncio.Event(),
            process=AsyncMock(),
            defer=AsyncMock(),
            purge_completed=AsyncMock(),
        )

    async def test_cleanup_timeouts_cannot_block_account_recovery(self):
        owner = self.owner(['pending'])
        synchronized = asyncio.Event()

        cleanup = _cleanup_timeouts_cannot_block_account_cleanup

        process = partial(
            _cleanup_timeouts_cannot_block_account_process,
            synchronized,
        )

        owner.purge_completed.side_effect = cleanup
        owner.process.side_effect = process
        maintenance = AccountMaintenance(owner)
        async with maintenance.lifespan():
            await asyncio.wait_for(synchronized.wait(), 0.5)
            await asyncio.sleep(0.025)
            self.assertGreater(owner.process.await_count, 1)
            self.assertGreater(owner.purge_completed.await_count, 1)
        self.assertTrue(owner.stop.is_set())

    async def test_slow_account_does_not_block_an_independent_account(self):
        owner = self.owner(['slow', 'fast'])
        fast = asyncio.Event()

        process = partial(_slow_account_does_not_block_process, fast)

        owner.process.side_effect = process
        running = asyncio.create_task(AccountMaintenance(owner).sweep())
        await asyncio.wait_for(fast.wait(), 0.05)
        await running
        owner.defer.assert_awaited_once_with('slow', 'TimeoutError')

    async def test_ownership_failure_isolated_to_one_operation(self):
        owner = self.owner(['offline', 'healthy'])
        owner.process.side_effect = [ConnectionError('pool'), True]
        await AccountMaintenance(owner).sweep()
        self.assertEqual(owner.process.await_count, 2)
        owner.defer.assert_awaited_once_with('offline', 'ConnectionError')

    async def test_shutdown_drains_active_sync_and_retention_tasks(self):
        owner = self.owner(['pending'])
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        process = partial(
            _shutdown_drains_active_sync_and_process,
            cancelled,
            entered,
        )

        owner.process.side_effect = process
        async with AccountMaintenance(owner).lifespan():
            await asyncio.wait_for(entered.wait(), 0.5)
        self.assertTrue(cancelled.is_set())

    async def test_invalid_progress_is_rejected_before_any_remote_effect(self):
        record = SimpleNamespace(
            kind='openwebui',
            sync_parts={'state': 'invalid', 'roles': 'pending'},
        )
        with patch.object(
            account_sync_steps.provider,
            'synchronize_openwebui_roles',
            AsyncMock(),
        ) as remote:
            with self.assertRaises(ValidationError):
                await AccountSynchronizer(None, record).step()
        remote.assert_not_awaited()
        with self.assertRaises(ValueError):
            AccountSyncPlan('unsupported', 'subject')
        with self.assertRaises(ValueError):
            AccountSyncPlan('openwebui', 'subject')
        with self.assertRaises(ValueError):
            await AccountSynchronizer(
                None,
                SimpleNamespace(kind='unsupported'),
            ).step()

    async def test_sync_budget_includes_selecting_pending_operations(self):
        from contextlib import asynccontextmanager

        owner = self.owner(['pending'])
        owner.settings.sync_budget_seconds = 0.02

        blocked_sessions = asynccontextmanager(
            _sync_budget_includes_selecting_pending_blocked_sessions,
        )

        owner.sessions = blocked_sessions
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(AccountMaintenance(owner).sweep(), 0.5)
        owner.process.assert_not_awaited()
