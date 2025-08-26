from __future__ import annotations

import unittest
from collections import defaultdict
from collections.abc import Awaitable
from collections.abc import Coroutine
from collections.abc import Iterable
from typing import Any
from typing import DefaultDict
from typing import TypeVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.local_notification_server import services as svc
from examples.local_notification_server.schemas import SiteNotifyRequest

# Generic type variable for async helpers
T = TypeVar('T')


class TestServices(unittest.TestCase):
    """
    Unit tests for notification services utilities.
    """

    def setUp(self) -> None:
        """Initialise a clean cache before each test.

        Ensures tests remain isolated and do not depend on call order.
        """
        # Reset in-memory cache to avoid cross-test interference
        svc._site_users_cache.clear()

    def test_decode_lang_token_map_basic(self) -> None:
        """It decodes raw Redis maps into a language-to-tokens mapping.

        Ensures empty language entries default to ``'en-GB'``.
        """
        raw = [
            {b't1': b'en-GB', b't2': b'zh'},
            {b't3': b'', b't4': b'ja'},
        ]
        got = svc._decode_lang_token_map(raw)
        # Convert to plain dict for assertion
        got_dict = {k: list(v) for k, v in got.items()}
        # empty -> en-GB
        self.assertEqual(sorted(got_dict['en-GB']), ['t1', 't3'])
        self.assertEqual(got_dict['zh'], ['t2'])
        self.assertEqual(got_dict['ja'], ['t4'])

    def test_get_lang_to_tokens_groups_by_lang(self) -> None:
        """It groups tokens by language using the Redis pipeline."""
        users = [MagicMock(id=1), MagicMock(id=2)]
        # Mock redis pipeline
        pipe = MagicMock()
        pipe.hgetall = MagicMock()
        pipe.execute = AsyncMock(
            return_value=[{b'a': b'en-GB'}, {b'b': b'zh'}],
        )
        rds = MagicMock()
        rds.pipeline.return_value = pipe

        got = self._run_async(svc._get_lang_to_tokens(users, rds))
        got_dict = {k: list(v) for k, v in got.items()}
        self.assertEqual(got_dict['en-GB'], ['a'])
        self.assertEqual(got_dict['zh'], ['b'])
        # Ensure hgetall called for each user
        self.assertEqual(pipe.hgetall.call_count, 2)

    def test_get_site_users_cached_miss_then_hit(self) -> None:
        """It returns DB results on first call then satisfies from cache."""
        # Prepare AsyncSession mock returning a site with users
        mock_user = MagicMock()
        mock_site = MagicMock(users=[mock_user])
        result = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.return_value = mock_site
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=result)

        # First call loads from DB
        users1 = self._run_async(
            svc.get_site_users_cached('SiteA', mock_session),
        )
        self.assertEqual(users1, [mock_user])

        # Modify session to fail if called again (to ensure cache used)
        mock_session.execute = AsyncMock(
            side_effect=AssertionError('should not query DB on cache hit'),
        )
        users2 = self._run_async(
            svc.get_site_users_cached('SiteA', mock_session),
        )
        self.assertEqual(users2, [mock_user])

    def test_get_site_users_cached_not_found(self) -> None:
        """It yields ``None`` when the site does not exist."""
        result = MagicMock()
        result.unique.return_value = result
        result.scalar_one_or_none.return_value = None
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=result)

        users = self._run_async(
            svc.get_site_users_cached('Missing', mock_session),
        )
        self.assertIsNone(users)

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_build_push_tasks_creates_tasks_per_lang(
        self, mock_send: AsyncMock,
    ) -> None:
        """
        It creates one task per language when batches are below 100 tokens.

        Args:
            mock_send (AsyncMock): The mocked send function.
        """
        # Two languages, each with tokens less than batch size -> 2 tasks
        lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(
            list,
            {
                'en-GB': ['t1', 't2'],
                'zh': ['t3'],
            },
        )
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'en': {'helmet': 1}},
            image_path=None,
            violation_id=123,
        )
        mock_send.return_value = True

        tasks = svc._build_push_tasks(req, lang_to_tokens)
        self.assertEqual(len(tasks), 2)
        # The tasks are coroutines produced by the AsyncMock
        # Trigger them to ensure they are awaitable
        results = self._run_async_many(tasks)
        self.assertEqual(results, [True, True])
        self.assertEqual(mock_send.await_count, 2)

    @patch(
        'examples.local_notification_server.services.'
        'send_fcm_notification_service',
        new_callable=AsyncMock,
    )
    def test_build_push_tasks_batches_over_100(
        self, mock_send: AsyncMock,
    ) -> None:
        """
        It splits tokens into batches of 100 for a given language.

        Args:
            mock_send (AsyncMock): The mocked send function.
        """
        # 205 tokens -> 3 batches at size 100
        tokens = [f"t{i}" for i in range(205)]
        lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(
            list,
            {
                'en-GB': tokens,
            },
        )
        req = SiteNotifyRequest(
            site='S1',
            stream_name='Cam1',
            body={'en': {'helmet': 1}},
            image_path=None,
            violation_id=None,
        )
        mock_send.return_value = True

        tasks = svc._build_push_tasks(req, lang_to_tokens)
        self.assertEqual(len(tasks), 3)
        _ = self._run_async_many(tasks)
        self.assertEqual(mock_send.await_count, 3)

    def test_execute_push_tasks_success_and_mix(self) -> None:
        """It returns mixed boolean results when tasks succeed/fail."""
        # Create two tasks: one True, one False
        t1 = AsyncMock(return_value=True)()
        t2 = AsyncMock(return_value=False)()
        ok, results, err = self._run_async(
            svc._execute_push_tasks([t1, t2], timeout=1.0),
        )
        self.assertTrue(ok)
        self.assertEqual(results, [True, False])
        self.assertIsNone(err)

    def test_execute_push_tasks_timeout(self) -> None:
        """It returns a timeout message when execution exceeds the limit."""
        with patch(
            'asyncio.wait_for',
            side_effect=__import__('asyncio').TimeoutError,
        ):
            ok, results, err = self._run_async(
                svc._execute_push_tasks([], timeout=0.01),
            )
            self.assertFalse(ok)
            self.assertIsNone(results)
            self.assertEqual(err, 'FCM notification sending timed out.')

    def test_execute_push_tasks_exception(self) -> None:
        """It returns a generic error indicator when an exception occurs."""
        with patch('asyncio.wait_for', side_effect=Exception('boom')):
            ok, results, err = self._run_async(
                svc._execute_push_tasks([], timeout=0.01),
            )
            self.assertFalse(ok)
            self.assertIsNone(results)
            self.assertEqual(err, 'internal_error')

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run a single awaitable to completion.

        Args:
            coro: The awaitable to execute.

        Returns:
            The awaited result, preserving the underlying type.
        """
        import asyncio
        return asyncio.run(coro)

    def _run_async_many(self, coros: Iterable[Awaitable[T]]) -> list[T]:
        """Run multiple awaitables concurrently and collect their results.

        Args:
            coros: An iterable of awaitables to execute.

        Returns:
            A list of results in the same order as the input awaitables.
        """
        import asyncio

        async def gatherer() -> list[T]:
            return await asyncio.gather(*coros)

        return asyncio.run(gatherer())


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.local_notification_server.services \
    --cov-report=term-missing \
    tests/examples/local_notification_server/services_test.py
"""
