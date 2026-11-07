from __future__ import annotations

import asyncio
import json
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from src.model_publication_status import report_loaded_model

pytestmark = pytest.mark.ml


async def _status_errors_do_not_break_run(redis, runtime, /):
    with patch(
        'src.model_publication_status.time.monotonic',
        return_value=100,
    ):
        for _ in range(10):
            await report_loaded_model(
                runtime,
                {
                    'catalog_revision': 'v',
                    'model_id': 'm',
                },
            )
    assert redis.get.await_count == 1
    redis.set.assert_not_awaited()


class _ModelStatusAnswersFreshPublicationChallengeAndCallbacks:
    metadata: Any
    redis: Any
    runtime: Any

    async def _call_run(self):
        with patch(
            'src.model_publication_status.time.monotonic',
            return_value=100,
        ):
            await report_loaded_model(self.runtime, self.metadata)
            await report_loaded_model(self.runtime, self.metadata)
        assert self.redis.set.await_count == 1
        status = json.loads(self.redis.set.call_args.args[1])
        assert status['publication'] == 'publication-one'
        assert status['catalog_revision'] == 'verified-revision'
        assert self.redis.set.call_args.kwargs['ex'] == 45
        self.redis.get.return_value = b'publication-two'
        with patch(
            'src.model_publication_status.time.monotonic',
            return_value=111,
        ):
            await report_loaded_model(self.runtime, self.metadata)
        assert (
            json.loads(self.redis.set.call_args.args[1])['publication']
            == 'publication-two'
        )


def test_model_status_answers_fresh_publication_challenge_and_is_throttled():
    _callbacks = _ModelStatusAnswersFreshPublicationChallengeAndCallbacks()
    _callbacks.redis = SimpleNamespace(
        get=AsyncMock(
            return_value=b'publication-one',
        ),
        set=AsyncMock(),
    )
    _callbacks.runtime = SimpleNamespace(
        cfg={'stream_id': 42},
        clock=SimpleNamespace(last_model_status_at=0),
        redis=SimpleNamespace(redis=_callbacks.redis),
    )
    _callbacks.metadata = {
        'model_id': 'detector',
        'model_version': 'v2',
        'catalog_revision': 'verified-revision',
        'model_sha256': 'b' * 64,
    }

    run = _callbacks._call_run

    asyncio.run(run())


def test_status_errors_do_not_break_detection_or_retry_on_every_frame():
    redis = SimpleNamespace(
        get=AsyncMock(
            side_effect=ConnectionError(),
        ),
        set=AsyncMock(),
    )
    runtime = SimpleNamespace(
        cfg={'stream_id': 42},
        clock=SimpleNamespace(last_model_status_at=0),
        redis=SimpleNamespace(redis=redis),
    )

    run = partial(_status_errors_do_not_break_run, redis, runtime)

    asyncio.run(run())
