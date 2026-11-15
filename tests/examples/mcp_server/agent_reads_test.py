from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.mcp_server.agent_reads.app import install
from examples.mcp_server.agent_reads.evidence import provenance
from examples.mcp_server.agent_reads.service import MAX_RESPONSE_BYTES
from examples.violation_records import violation_query_service
from examples.violation_records import violation_review_service

HEADERS = {
    'Accept': 'application/json,text/event-stream',
    'MCP-Protocol-Version': '2025-11-25',
}


async def _database() -> AsyncIterator[SimpleNamespace]:
    database = SimpleNamespace(closed=False)
    try:
        yield database
    finally:
        database.closed = True


def _credentials(request: Request) -> JwtAuthorizationCredentials:
    token = request.headers.get('Authorization')
    users = {
        'Bearer alice-token': ('alice', 1),
        'Bearer bob-token': ('bob', 2),
    }
    if token not in users:
        raise HTTPException(401, 'Missing personal bearer')
    name, identifier = users[token]
    return JwtAuthorizationCredentials(
        subject=cast(
            AccessTokenSubject,
            {'username': name, 'user_id': identifier, 'role': 'user'},
        ),
        payload={'sub': name + '-oidc-subject'},
        token=token.removeprefix('Bearer '),
    )


class AgentReadsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = FastAPI()
        install(self.app)
        self.app.dependency_overrides[jwt_access] = _credentials
        self.app.dependency_overrides[get_db] = _database
        self.client = self.enterContext(
            TestClient(self.app, base_url='https://construction.example'),
        )
        self.seen: list[dict[str, Any]] = []

    def rpc(
        self,
        method: str,
        params: dict[str, Any],
        bearer: str = 'alice-token',
    ) -> Any:
        return self.client.post(
            '/mcp',
            headers={**HEADERS, 'Authorization': 'Bearer ' + bearer},
            json={
                'jsonrpc': '2.0', 'id': 1,
                'method': method, 'params': params,
            },
        )

    def tool(
        self,
        name: str,
        arguments: dict[str, Any],
        bearer: str = 'alice-token',
    ) -> dict[str, Any]:
        response = self.rpc(
            'tools/call', {'name': name, 'arguments': arguments}, bearer,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['result']

    async def sites(self, db: Any, credentials: Any) -> list[dict[str, Any]]:
        self.assertFalse(db.closed)
        self.seen.append(
            {'user': credentials.subject['username'], 'database': db},
        )
        return [
            {
                'id': credentials.subject['user_id'],
                'name': credentials.subject['username'] + ' site',
                'token': 'upstream-private-credential',
                'image_url': 'https://private.example/image?token=private',
            },
        ]

    async def page(
        self, request: Request, db: Any, credentials: Any, **kwargs: Any,
    ) -> dict[str, Any]:
        self.assertFalse(db.closed)
        self.seen.append(
            {
                'user': credentials.subject['username'],
                'host': request.url.hostname,
                'scheme': request.url.scheme,
                'filters': kwargs['filters'],
                'limit': kwargs['limit'],
            },
        )
        return {
            'items': [
                {
                    'id': 42,
                    'site_name': 'Authorized site',
                    'stream_name': 'camera-1',
                    'detection_time': '2026-11-01T10:00:00+08:00',
                    'image_url': 'private-image-url',
                },
            ],
            'has_more': True,
            'next_cursor': 'opaque-next-page',
            'bearer': 'upstream-secret',
        }

    async def delayed_sites(
        self, db: Any, credentials: Any,
    ) -> list[dict[str, Any]]:
        await asyncio.sleep(0.02)
        return await self.sites(db, credentials)

    def test_real_sdk_discovery_owns_tools_schemas_and_read_only_annotations(
        self,
    ) -> None:
        initialized = self.rpc(
            'initialize',
            {
                'protocolVersion': '2025-11-25',
                'capabilities': {},
                'clientInfo': {'name': 'generic-agent', 'version': '1'},
            },
        )
        self.assertEqual(initialized.status_code, 200)
        response = self.rpc('tools/list', {})
        self.assertEqual(response.status_code, 200)
        tools = response.json()['result']['tools']
        self.assertEqual(
            {tool['name'] for tool in tools},
            {
                'construction_my_sites',
                'construction_search_violations',
                'construction_get_violation',
            },
        )
        for tool in tools:
            self.assertTrue(tool['annotations']['readOnlyHint'])
            self.assertFalse(tool['annotations']['destructiveHint'])
            schema = tool['inputSchema']
            self.assertFalse(schema['additionalProperties'])
            self.assertFalse(
                set(schema['properties'])
                & {'owner', 'subject', 'token', 'runtime', 'ctx', 'url'},
            )
        search = next(
            tool for tool in tools
            if tool['name'] == 'construction_search_violations'
        )
        self.assertEqual(
            search['inputSchema']
            ['properties']['limit']['maximum'], 100,
        )

    def test_anonymous_requests_are_rejected_before_domain_query(self) -> None:
        del self.app.dependency_overrides[jwt_access]
        with patch.object(
            violation_query_service, 'get_my_sites', AsyncMock(),
        ) as sites:
            response = self.client.post(
                '/mcp',
                headers=HEADERS,
                json={'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'},
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers['www-authenticate'], 'Bearer')
        sites.assert_not_awaited()

    def test_each_call_uses_its_own_user_and_live_database(self) -> None:
        with patch.object(
            violation_query_service, 'get_my_sites', self.sites,
        ):
            alice = self.tool('construction_my_sites', {})
            bob = self.tool('construction_my_sites', {}, 'bob-token')
        self.assertEqual([row['user'] for row in self.seen], ['alice', 'bob'])
        self.assertTrue(all(row['database'].closed for row in self.seen))
        self.assertEqual(
            alice['structuredContent']['data'], [
                {'id': 1, 'name': 'alice site'},
            ],
        )
        self.assertEqual(
            bob['structuredContent']['data'], [{'id': 2, 'name': 'bob site'}],
        )
        self.assertEqual(
            alice['structuredContent']
            ['provenance']['site_ids'], [1],
        )
        self.assertNotIn('upstream-private-credential', json.dumps(alice))
        self.assertNotIn('alice-token', json.dumps(alice))

    def test_search_preserves_host_filters_and_pagination(self) -> None:
        with patch.object(
            violation_query_service, 'get_violations', self.page,
        ):
            result = self.tool(
                'construction_search_violations',
                {'site_id': 4, 'keyword': 'hardhat', 'limit': 1},
            )
        self.assertFalse(result['isError'])
        content = result['structuredContent']
        self.assertEqual(content['source'], 'https://construction.example/mcp')
        self.assertEqual(content['data']['next_cursor'], 'opaque-next-page')
        self.assertTrue(content['data']['has_more'])
        self.assertNotIn('total', content['data'])
        self.assertEqual(content['provenance']['returned_count'], 1)
        self.assertTrue(content['provenance']['has_more'])
        self.assertEqual(content['provenance']['query']['site_id'], 4)
        self.assertEqual(self.seen[0]['host'], 'construction.example')
        self.assertEqual(self.seen[0]['scheme'], 'https')
        self.assertEqual(self.seen[0]['filters'].site_id, 4)
        self.assertNotIn('private-image-url', json.dumps(content))
        self.assertNotIn('upstream-secret', json.dumps(content))

    def test_concurrent_users_keep_separate_request_identity(self) -> None:
        with (
            patch.object(
                violation_query_service, 'get_my_sites', self.delayed_sites,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            alice = executor.submit(
                self.tool, 'construction_my_sites', {}, 'alice-token',
            )
            bob = executor.submit(
                self.tool, 'construction_my_sites', {}, 'bob-token',
            )
            results = [alice.result(), bob.result()]
        self.assertEqual(
            [
                result['structuredContent']['provenance']['site_ids']
                for result in results
            ],
            [[1], [2]],
        )
        self.assertCountEqual(
            [row['user'] for row in self.seen], ['alice', 'bob'],
        )

    def test_denied_detail_has_safe_error_without_fallback(self) -> None:
        with patch.object(
            violation_review_service,
            'get_single_violation',
            AsyncMock(side_effect=HTTPException(403, 'private-error-detail')),
        ) as read:
            result = self.tool(
                'construction_get_violation', {'violation_id': 42},
            )
        self.assertTrue(result['isError'])
        self.assertIn('無權', result['content'][0]['text'])
        self.assertNotIn('private-error-detail', json.dumps(result))
        read.assert_awaited_once()
        assert read.await_args is not None
        self.assertEqual(read.await_args.args[3].subject['username'], 'alice')

    def test_invalid_arguments_do_not_reach_domain_services(self) -> None:
        cases = (
            ('construction_my_sites', {'owner': 'bob'}),
            ('construction_search_violations', {'limit': 101}),
            ('construction_search_violations', {'site_id': True}),
            ('construction_search_violations', {'start_time': '2026-11-01'}),
            ('construction_search_violations', {'token': 'forged-token'}),
            ('construction_get_violation', {'violation_id': 0}),
        )
        with (
            patch.object(
                violation_query_service, 'get_my_sites', AsyncMock(),
            ) as sites,
            patch.object(
                violation_query_service, 'get_violations', AsyncMock(),
            ) as search,
            patch.object(
                violation_review_service, 'get_single_violation', AsyncMock(),
            ) as detail,
        ):
            for operation, arguments in cases:
                with self.subTest(operation=operation, arguments=arguments):
                    result = self.tool(operation, arguments)
                    self.assertTrue(result['isError'])
        sites.assert_not_awaited()
        search.assert_not_awaited()
        detail.assert_not_awaited()

    def test_detail_projection_and_metadata_exclude_media(self) -> None:
        with patch.object(
            violation_review_service,
            'get_single_violation',
            AsyncMock(
                return_value={
                    'id': 42,
                    'site_name': 'Site A',
                    'stream_name': 'Camera A',
                    'detection_time': '2026-11-01T10:00:00+08:00',
                    'created_at': '2026-11-01T10:00:00+08:00',
                    'image_url': 'secret-media-url',
                    'image_path': '/private/source/image',
                },
            ),
        ):
            result = self.tool(
                'construction_get_violation', {'violation_id': 42},
            )
        self.assertFalse(result['isError'])
        self.assertEqual(
            result['structuredContent']['provenance']['violation_ids'], [42],
        )
        self.assertNotIn('secret-media-url', json.dumps(result))
        self.assertNotIn('/private/source/image', json.dumps(result))

    def test_malformed_or_oversized_data_is_not_success(self) -> None:
        for value in (
            [{'name': 'missing id'}],
            [{'id': 1, 'name': 'x' * (MAX_RESPONSE_BYTES + 1)}],
        ):
            with (
                self.subTest(value_size=len(str(value))),
                patch.object(
                    violation_query_service,
                    'get_my_sites',
                    AsyncMock(return_value=value),
                ),
            ):
                result = self.tool('construction_my_sites', {})
                self.assertTrue(result['isError'])

    def test_large_identifier_list_marks_provenance_truncation(self) -> None:
        envelope = {
            'source': 'https://construction.example/mcp',
            'operation': 'construction_my_sites',
            'queried_at': '2026-11-01T00:00:00+00:00',
            'query': {},
            'data': [{'id': index} for index in range(1, 102)],
            'limitations': ['read scope'],
        }
        evidence = provenance(envelope)
        self.assertEqual(evidence['returned_count'], 101)
        self.assertEqual(len(evidence['site_ids']), 100)
        self.assertTrue(evidence['ids_truncated'])

    def test_install_is_idempotent(self) -> None:
        lifespan = self.app.router.lifespan_context
        install(self.app)
        self.assertIs(self.app.router.lifespan_context, lifespan)
        self.assertEqual(
            sum(
                getattr(route, 'path', None) ==
                '/mcp' for route in self.app.routes
            ),
            1,
        )
