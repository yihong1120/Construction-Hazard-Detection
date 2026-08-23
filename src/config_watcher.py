from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from collections.abc import Coroutine
from typing import Any

from watchdog.events import FileSystemEvent
from watchdog.events import FileSystemEventHandler


class FileEventHandler(FileSystemEventHandler):
    """Schedule at most one in-flight reload for a watched configuration
    file."""

    def __init__(
        self,
        file_path: str,
        callback: Callable[[], Coroutine[Any, Any, None]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Perform init.

        Args:
            file_path: Value used by this callable.
            callback: Value used by this callable.
            loop: Value used by this callable.
        """
        self.file_path = os.path.abspath(file_path)
        self.callback = callback
        self.loop = loop
        self.logger = logging.getLogger(__name__)
        self._pending: object | None = None

    def on_modified(self, event: FileSystemEvent) -> None:
        """Coalesce duplicate watchdog events and schedule a safe reload."""
        if os.path.abspath(os.fsdecode(event.src_path)) != self.file_path:
            return
        pending = self._pending
        if (
            pending is not None
            and not getattr(pending, 'done', lambda: True)()
        ):
            self.logger.debug(
                'Configuration reload already queued path=%s',
                self.file_path,
            )
            return
        self.logger.info('Configuration file modified path=%s', self.file_path)
        self._pending = asyncio.run_coroutine_threadsafe(
            self.callback(),
            self.loop,
        )
