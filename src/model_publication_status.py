from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from examples.shared.model_status import model_challenge_key
from examples.shared.model_status import model_status_key

if TYPE_CHECKING:
    from src.stream_runtime_state import StreamRuntime

logger = logging.getLogger(__name__)


async def report_loaded_model(runtime: StreamRuntime, metadata: dict) -> None:
    stream_id = runtime.cfg.get('stream_id')
    if stream_id is None or not metadata.get('catalog_revision'):
        return
    now = time.monotonic()
    if now - runtime.clock.last_model_status_at < 10:
        return
    runtime.clock.last_model_status_at = now
    try:
        async with asyncio.timeout(1):
            challenge = await runtime.redis.redis.get(
                model_challenge_key(metadata['model_id']),
            )
            await runtime.redis.redis.set(
                model_status_key(stream_id),
                json.dumps({
                    'model_id': metadata['model_id'],
                    'model_version': metadata['model_version'],
                    'catalog_revision': metadata['catalog_revision'],
                    'model_sha256': metadata['model_sha256'],
                    'publication': challenge.decode()
                    if isinstance(challenge, bytes) else challenge,
                }).encode(),
                ex=45,
            )
    except Exception:
        logger.debug('Model load status could not be published', exc_info=True)
