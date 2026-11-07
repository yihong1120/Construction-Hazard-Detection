from __future__ import annotations

import asyncio
import json
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from examples.shared.model_status import model_status_key
from scripts import model_publication as module


async def _membership_is_rechecked_after_acknowledgements_yield_once(
    sleep,
    /,
    *args,
):
    await sleep(0)


async def _first_activation_initializes_a_challenge_yield_once(
    sleep,
    /,
    *args,
):
    await sleep(0)


class PublicationMembershipTest(unittest.IsolatedAsyncioTestCase):
    def publication(self):
        publication = module.ModelPublication()
        publication.challenge = 'challenge'
        self.ack = json.dumps(
            {
                'model_id': 'detector',
                'catalog_revision': 'revision',
                'publication': 'challenge',
            },
        )
        return publication

    async def test_newly_active_stream_is_required_before_success(self):
        publication = self.publication()
        publication._active_streams = AsyncMock(
            return_value={1: 'detector', 2: 'detector'},
        )
        publication.redis = SimpleNamespace(
            mget=AsyncMock(return_value=[self.ack, None]),
        )
        with self.assertRaises(TimeoutError):
            await publication._verify({'detector': 'revision'}, 0.03)
        publication.redis.mget.assert_awaited_once_with(
            [model_status_key(1), model_status_key(2)],
        )

    async def test_membership_is_rechecked_after_acknowledgements(self):
        publication = self.publication()
        one, two = {1: 'detector'}, {1: 'detector', 2: 'detector'}
        publication._active_streams = AsyncMock(
            side_effect=[one, two, two, two],
        )
        publication.redis = SimpleNamespace(
            mget=AsyncMock(side_effect=[[self.ack], [self.ack, self.ack]]),
        )
        sleep = asyncio.sleep

        yield_once = partial(
            _membership_is_rechecked_after_acknowledgements_yield_once,
            sleep,
        )

        with patch.object(module.asyncio, 'sleep', yield_once):
            await publication._verify({'detector': 'revision'}, 0.2)
        self.assertEqual(publication.redis.mget.await_count, 2)
        self.assertEqual(
            publication.redis.mget.await_args.args[0],
            [model_status_key(1), model_status_key(2)],
        )

    async def test_first_activation_initializes_a_challenge(
        self,
    ):
        publication = self.publication()
        publication._active_streams = AsyncMock(
            side_effect=[
                {},
                {2: 'detector'},
                {2: 'detector'},
                {2: 'detector'},
            ],
        )
        redis = SimpleNamespace(
            mget=AsyncMock(return_value=[self.ack]),
            ping=AsyncMock(),
        )
        sleep = asyncio.sleep

        yield_once = partial(
            _first_activation_initializes_a_challenge_yield_once,
            sleep,
        )

        with patch.object(
            module,
            'RedisManager',
            return_value=SimpleNamespace(redis=redis),
        ), patch.object(
            publication,
            '_prepare_challenge',
            new=AsyncMock(),
        ) as challenge, patch.object(
            module.asyncio,
            'sleep',
            yield_once,
        ):
            await publication._verify({'detector': 'revision'}, 0.2)
        challenge.assert_awaited_once()
        redis.ping.assert_awaited_once()
