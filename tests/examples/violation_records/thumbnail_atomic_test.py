from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from examples.shared.image_work import close_image_work
from examples.violation_records import media_service as module
from tests.image_fixtures_test import image_bytes


def _failed_encode_keeps_cache_and_broken(image, path, *args, **kwargs):
    Path(path).write_bytes(b'new partial')
    raise OSError('encoder failed')


class _WaitersShareWorkAndCancelIndependentlyCallbacks:
    calls: Any
    entered: Any
    original: Any
    release: Any

    def _call_generate(self, *args):
        self.calls.append(1)
        self.entered.set()
        self.release.wait(3)
        return self.original(*args)


class ThumbnailAtomicTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / 'image.png'
        self.source.write_bytes(image_bytes())
        self.addAsyncCleanup(close_image_work)

    async def test_failed_encode_keeps_cache_and_retry_repairs_it(
        self,
    ):
        target = module._thumbnail_cache_path(self.source, self.root)
        target.parent.mkdir(parents=True)
        target.write_bytes(b'legacy partial')

        broken = _failed_encode_keeps_cache_and_broken

        with patch.object(Image.Image, 'save', broken):
            with self.assertRaises(HTTPException):
                await module.ensure_thumbnail(self.source, self.root)
        self.assertEqual(target.read_bytes(), b'legacy partial')
        self.assertEqual(list(target.parent.glob('*.tmp')), [])
        await module.ensure_thumbnail(self.source, self.root)
        with Image.open(target) as image:
            image.load()
            self.assertEqual(image.format, 'JPEG')

    async def test_waiters_share_work_and_cancel_independently(
        self,
    ):
        _callbacks = _WaitersShareWorkAndCancelIndependentlyCallbacks()
        _callbacks.entered, _callbacks.release = (
            threading.Event(),
            threading.Event(),
        )
        _callbacks.original = module._generate_thumbnail_sync
        _callbacks.calls = []

        generate = _callbacks._call_generate

        with patch.object(module, '_generate_thumbnail_sync', generate):
            first = asyncio.create_task(
                module.ensure_thumbnail(self.source, self.root),
            )
            async with asyncio.timeout(1):
                while not _callbacks.entered.is_set():
                    await asyncio.sleep(0.005)
            second = asyncio.create_task(
                module.ensure_thumbnail(self.source, self.root),
            )
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            _callbacks.release.set()
            target = await second
        self.assertEqual(_callbacks.calls, [1])
        self.assertTrue(target.is_file())
        self.assertFalse(module._thumbnail_jobs)

    async def test_different_source_extensions_have_distinct_cache_keys(self):
        other = self.root / 'image.jpg'
        other.write_bytes(image_bytes('blue', 'JPEG'))
        first, second = await asyncio.gather(
            module.ensure_thumbnail(self.source, self.root),
            module.ensure_thumbnail(other, self.root),
        )
        self.assertNotEqual(first, second)
        with Image.open(first) as red, Image.open(second) as blue:
            self.assertNotEqual(red.getpixel((0, 0)), blue.getpixel((0, 0)))
