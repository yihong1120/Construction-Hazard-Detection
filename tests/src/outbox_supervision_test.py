from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

from src.violation_delivery import OutboxDispatcher


class _ScanFailureRecoversWithoutCameraRestartCallbacks:
    discover: Any
    first: Any
    loop: Any
    recovered: Any

    def _call_scan(self):

        if self.first:
            self.first = False
            raise OSError('temporary disk failure')
        self.discover()
        self.loop.call_soon_threadsafe(self.recovered.set)


class _ConcurrentCloseDrainsOnceAndPreventsEarlyCallbacks:
    cancellations: Any
    cleaning: Any
    entered: Any
    release: Any

    async def _call_run(self):
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellations.append(1)
            self.cleaning.set()
            await self.release.wait()
            raise


class OutboxSupervisionTest(unittest.IsolatedAsyncioTestCase):
    async def test_scan_failure_recovers_without_camera_restart(
        self,
    ):
        _callbacks = _ScanFailureRecoversWithoutCameraRestartCallbacks()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with patch.dict(
            os.environ,
            {
                'VIOLATION_OUTBOX_DIRECTORY': directory.name,
                'VIOLATION_OUTBOX_RECOVERY_SECONDS': '.01',
            },
        ):
            dispatcher = OutboxDispatcher()
        _callbacks.first = True
        _callbacks.recovered = asyncio.Event()
        _callbacks.loop = asyncio.get_running_loop()
        _callbacks.discover = dispatcher.directory.discover

        scan = _callbacks._call_scan

        resources = AsyncMock()
        with patch(
            'src.violation_delivery.TokenManager',
            return_value=resources,
        ), patch(
            'src.violation_delivery.ViolationSender',
            return_value=resources,
        ), patch(
            'src.violation_delivery.FCMSender',
            return_value=resources,
        ), patch.object(
            dispatcher.directory,
            'discover',
            scan,
        ):
            dispatcher.start()
            try:
                await asyncio.wait_for(_callbacks.recovered.wait(), 1)
                async with asyncio.timeout(1):
                    while dispatcher.health['last_poll'] is None:
                        await asyncio.sleep(0.005)
                self.assertTrue(dispatcher.health['running'])
                self.assertEqual(dispatcher.health['failures'], 1)
                self.assertIsNone(dispatcher.health['last_error'])
            finally:
                await dispatcher.close()
        self.assertFalse(dispatcher.health['running'])

    async def test_repeated_failure_uses_backoff_and_close_interrupts_retry(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                'VIOLATION_OUTBOX_DIRECTORY': directory,
                'VIOLATION_OUTBOX_RECOVERY_SECONDS': '.01',
            },
        ):
            dispatcher = OutboxDispatcher()
            broken = AsyncMock(side_effect=OSError('unavailable'))
            with patch.object(dispatcher, 'run', broken):
                dispatcher.start()
                async with asyncio.timeout(1):
                    while dispatcher.health['failures'] < 3:
                        await asyncio.sleep(0.005)
                self.assertEqual(dispatcher.health['last_error'], 'OSError')
                self.assertGreaterEqual(dispatcher._recovery_attempts, 3)
                await asyncio.wait_for(dispatcher.close(), 0.2)
            self.assertFalse(dispatcher.health['running'])

    async def test_unexpected_clean_exit_is_supervised(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                'VIOLATION_OUTBOX_DIRECTORY': directory,
                'VIOLATION_OUTBOX_RECOVERY_SECONDS': '.01',
            },
        ):
            dispatcher = OutboxDispatcher()
            with patch.object(dispatcher, 'run', AsyncMock()):
                dispatcher.start()
                async with asyncio.timeout(1):
                    while dispatcher.health['failures'] < 2:
                        await asyncio.sleep(0.005)
                self.assertEqual(
                    dispatcher.health['last_error'],
                    'RuntimeError',
                )
                await dispatcher.close()

    async def test_concurrent_close_drains_once_and_prevents_early_restart(
        self,
    ):
        _callbacks = _ConcurrentCloseDrainsOnceAndPreventsEarlyCallbacks()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                'VIOLATION_OUTBOX_DIRECTORY': directory,
            },
        ):
            dispatcher = OutboxDispatcher()
            _callbacks.entered, _callbacks.cleaning, _callbacks.release = (
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
            )
            _callbacks.cancellations = []

            run = _callbacks._call_run

            with patch.object(dispatcher, 'run', run):
                dispatcher.start()
                await _callbacks.entered.wait()
                first = asyncio.create_task(dispatcher.close())
                await _callbacks.cleaning.wait()
                with self.assertRaises(RuntimeError):
                    dispatcher.start()
                second = asyncio.create_task(dispatcher.close())
                first.cancel()
                await asyncio.sleep(0)
                self.assertFalse(first.done())
                _callbacks.release.set()
                await second
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertEqual(_callbacks.cancellations, [1])
                self.assertIsNone(dispatcher.task)
