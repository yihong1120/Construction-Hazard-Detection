from __future__ import annotations

from functools import partial
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from tests.examples.auth.session_store_test import FakePipeline
from tests.examples.auth.session_store_test import FakeRedis


def _spy_redis_pipeline(redis, /, transaction=True):
    pipe = FakePipeline(redis)
    pipe.queued = True
    return pipe


def spy_redis():
    backend = FakeRedis()
    redis = MagicMock()
    for name in ('data', 'ttls', 'sets', 'zsets'):
        setattr(redis, name, getattr(backend, name))
    for name in dir(FakeRedis):
        if not name.startswith('_') and name != 'pipeline':
            setattr(redis, name, AsyncMock(wraps=getattr(backend, name)))

    pipeline = partial(_spy_redis_pipeline, redis)

    redis.pipeline.side_effect = pipeline
    return redis


def watched_pipeline():
    pipeline = MagicMock()
    pipeline.__aenter__ = AsyncMock(return_value=pipeline)
    pipeline.__aexit__ = AsyncMock(return_value=None)
    pipeline.watch = AsyncMock()
    pipeline.smembers = AsyncMock(return_value=set())
    pipeline.execute = AsyncMock()
    return pipeline
