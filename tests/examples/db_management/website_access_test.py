from __future__ import annotations

import json
import unittest
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from deploy.website_access.agent import apply_once
from deploy.website_access.agent import render
from deploy.website_access.deploy import gateway_content
from deploy.website_access.deploy import patched_code
from examples.auth.database import get_db
from examples.db_management.deps import get_current_user
from examples.db_management.routers.website_access import router
from examples.db_management.services.website_access import ip_allowed
from examples.db_management.services.website_access import website_hosts
from examples.db_management.website_access_models import WebsiteAccessAudit
from examples.db_management.website_access_models import WebsiteAccessPolicy
from examples.db_management.website_access_models import WebsiteAccessRule


async def _asyncSetUp_db_override(testcase, /):
    async with testcase.sessions() as db:
        yield db


def _asyncSetUp_get_user(testcase, /):
    return testcase.user


class WebsiteAccessApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine('sqlite+aiosqlite://')
        async with self.engine.begin() as c:
            await c.execute(
                text('CREATE TABLE users (id INTEGER PRIMARY KEY)'),
            )
            await c.execute(text('INSERT INTO users VALUES (1)'))
            for model in (
                WebsiteAccessPolicy,
                WebsiteAccessRule,
                WebsiteAccessAudit,
            ):
                await c.run_sync(model.__table__.create)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.sessions() as db:
            db.add_all(
                [
                    WebsiteAccessPolicy(site_key=key)
                    for key in ('main', 'ai', 'wiki')
                ],
            )
            await db.commit()
        self.user = SimpleNamespace(id=1, username='ChangDar', role='admin')
        app = FastAPI()
        app.include_router(router)

        db_override = partial(_asyncSetUp_db_override, self)

        app.dependency_overrides[get_db] = db_override
        app.dependency_overrides[get_current_user] = partial(
            _asyncSetUp_get_user,
            self,
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=app,
                client=('203.0.113.5', 1234),
            ),
            base_url='http://test',
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.engine.dispose()

    async def add(
        self,
        site='ai',
        network='203.0.113.5',
        version=0,
        label='Office',
    ):
        return await self.client.post(
            '/admin/website-access/' + site + '/rules',
            json={
                'network': network,
                'label': label,
                'expected_version': version,
            },
        )

    async def test_crud_audit_and_site_isolation(self):
        result = await self.client.get('/admin/website-access')
        self.assertEqual(len(result.json()), 3)
        created = await self.add()
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(
            created.json()['rules'][0]['network'],
            '203.0.113.5/32',
        )
        rule_id = created.json()['rules'][0]['id']
        wrong_site = await self.client.delete(
            '/admin/website-access/wiki/rules/' + rule_id,
            params={'expected_version': 0},
        )
        self.assertEqual(wrong_site.status_code, 404)
        updated = await self.client.put(
            '/admin/website-access/ai/rules/' + rule_id,
            json={
                'network': '203.0.113.0/24',
                'label': 'Office LAN',
                'expected_version': 1,
            },
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        deleted = await self.client.delete(
            '/admin/website-access/ai/rules/' + rule_id,
            params={'expected_version': 2},
        )
        self.assertEqual(deleted.json()['rules'], [])
        audits = await self.client.get(
            '/admin/website-access/audit?site_key=ai',
        )
        self.assertEqual(len(audits.json()), 3)
        self.assertEqual(audits.json()[0]['action'], 'rule_deleted')
        self.assertEqual(audits.json()[0]['actor_ip'], '203.0.113.5')
        self.assertEqual(
            (
                await self.client.get('/admin/website-access/audit?limit=201')
            ).status_code,
            422,
        )

    async def test_enabled_policy_cannot_lock_out_operator(self):
        empty = await self.client.put(
            '/admin/website-access/main',
            json={'restricted': True, 'expected_version': 0},
        )
        self.assertEqual(empty.status_code, 409)
        self.assertEqual(
            (await self.client.get('/admin/website-access/audit')).json(),
            [],
        )
        created = (await self.add(site='main')).json()
        enabled = await self.client.put(
            '/admin/website-access/main',
            json={'restricted': True, 'expected_version': 1},
        )
        self.assertEqual(enabled.status_code, 200, enabled.text)
        rule_id = created['rules'][0]['id']
        deleted = await self.client.delete(
            '/admin/website-access/main/rules/' + rule_id,
            params={'expected_version': 2},
        )
        self.assertEqual(deleted.status_code, 409)
        disabled = await self.client.put(
            '/admin/website-access/main/rules/' + rule_id,
            json={
                'network': '203.0.113.5',
                'label': 'Office',
                'enabled': False,
                'expected_version': 2,
            },
        )
        self.assertEqual(disabled.status_code, 409)
        moved = await self.client.put(
            '/admin/website-access/main/rules/' + rule_id,
            json={
                'network': '192.0.2.1',
                'label': 'Other office',
                'expected_version': 2,
            },
        )
        self.assertEqual(moved.status_code, 409)
        policy = next(
            p
            for p in (await self.client.get('/admin/website-access')).json()
            if p['site_key'] == 'main'
        )
        self.assertTrue(policy['restricted'])
        self.assertEqual(policy['version'], 2)
        self.assertEqual(policy['rules'][0]['network'], '203.0.113.5/32')

    async def test_conflicts_validation_and_host_allowlist(self):
        await self.add()
        duplicate = await self.add(version=1)
        self.assertEqual(duplicate.status_code, 409)
        stale = await self.add(network='192.0.2.1', version=0)
        self.assertEqual(stale.status_code, 409)
        unknown = await self.add(site='arbitrary.example.com')
        self.assertEqual(unknown.status_code, 422)
        for network in (
            '0.0.0.0/0',
            '::/0',
            'example.com',
            '1.1.1.1; deny all;',
            'fe80::1%en0',
            '::ffff:192.0.2.1',
        ):
            with self.subTest(network=network):
                self.assertEqual(
                    (await self.add(network=network, version=1)).status_code,
                    422,
                )

    async def test_non_global_admin_cannot_read_or_mutate(self):
        for role in ('user', 'admin'):
            self.user = SimpleNamespace(id=2, username='Other', role=role)
            for method, path, body in [
                ('GET', '', None),
                ('GET', '/audit', None),
                ('GET', '/gateway', None),
                ('GET', '/context', None),
                ('PUT', '/ai', {'restricted': False, 'expected_version': 0}),
            ]:
                result = await self.client.request(
                    method,
                    '/admin/website-access' + path,
                    json=body,
                )
                self.assertEqual(result.status_code, 403)
            self.assertEqual((await self.add()).status_code, 403)

    async def test_context_ignores_forged_headers_and_gateway_is_explicit(
        self,
    ):
        response = await self.client.get(
            '/admin/website-access/context',
            headers={'X-Forwarded-For': '1.2.3.4', 'X-Real-IP': '1.2.3.4'},
        )
        self.assertEqual(response.json()['client_ip'], '203.0.113.5')
        with patch.dict(
            'os.environ',
            {'WEBSITE_ACCESS_STATUS_FILE': '/nonexistent/receipt'},
        ):
            self.assertFalse(
                (
                    await self.client.get('/admin/website-access/gateway')
                ).json()['online'],
            )

    async def test_gateway_receipt_requires_fresh_matching_hosts_and_versions(
        self,
    ):
        with TemporaryDirectory() as directory:
            receipt_path = Path(directory) / 'status.json'
            receipt = {
                'checked_at': datetime.now(timezone.utc).isoformat(),
                'hosts': website_hosts(),
                'versions': {'main': 0, 'ai': 0, 'wiki': 0},
            }
            with patch.dict(
                'os.environ',
                {'WEBSITE_ACCESS_STATUS_FILE': str(receipt_path)},
            ):
                receipt_path.write_text(json.dumps(receipt))
                response = await self.client.get(
                    '/admin/website-access/gateway',
                )
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()['online'])
                self.assertEqual(
                    response.json()['versions'],
                    receipt['versions'],
                )

                stale = dict(receipt)
                stale['checked_at'] = (
                    datetime.now(timezone.utc) - timedelta(minutes=1)
                ).isoformat()
                receipt_path.write_text(json.dumps(stale))
                self.assertFalse(
                    (
                        await self.client.get('/admin/website-access/gateway')
                    ).json()['online'],
                )

                wrong_hosts = dict(receipt)
                wrong_hosts['hosts'] = dict(receipt['hosts'], ai='other.test')
                receipt_path.write_text(json.dumps(wrong_hosts))
                self.assertFalse(
                    (
                        await self.client.get('/admin/website-access/gateway')
                    ).json()['online'],
                )

                invalid_versions = dict(receipt)
                invalid_versions['versions'] = dict(
                    receipt['versions'],
                    ai=True,
                )
                receipt_path.write_text(json.dumps(invalid_versions))
                self.assertEqual(
                    (
                        await self.client.get('/admin/website-access/gateway')
                    ).json(),
                    {
                        'online': False,
                        'checked_at': None,
                        'versions': {},
                    },
                )

    async def test_missing_catalog_does_not_silently_open(self):
        async with self.sessions() as db:
            row = (
                await db.execute(
                    select(WebsiteAccessPolicy).where(
                        WebsiteAccessPolicy.site_key == 'wiki',
                    ),
                )
            ).scalar_one()
            await db.delete(row)
            await db.commit()
        response = await self.client.get('/admin/website-access')
        self.assertEqual(response.status_code, 503)


class WebsiteGatewayTest(unittest.TestCase):
    def data(self):
        return {
            'hosts': {
                'main': 'main.test',
                'ai': 'ai.test',
                'wiki': 'wiki.test',
            },
            'policies': [
                {
                    'site_key': key,
                    'restricted': key == 'ai',
                    'version': 1,
                    'rules': [{'network': '203.0.113.0/24', 'enabled': True}],
                }
                for key in ('main', 'ai', 'wiki')
            ],
        }

    def test_matching_ipv4_ipv6_and_disabled_rules(self):
        policy = self.data()['policies'][1]
        self.assertTrue(ip_allowed(policy, '203.0.113.5'))
        self.assertTrue(ip_allowed(policy, '::ffff:203.0.113.5'))
        self.assertFalse(ip_allowed(policy, '203.0.114.5'))
        self.assertFalse(ip_allowed(policy, '2001:db8::1'))
        policy['rules'].append({'network': '2001:db8::/64', 'enabled': True})
        self.assertTrue(ip_allowed(policy, '2001:db8::1'))
        policy['rules'][0]['enabled'] = False
        self.assertFalse(ip_allowed(policy, '203.0.113.5'))

    def test_generated_rules_are_numeric_and_complete(self):
        rendered, versions = render(self.data())
        self.assertIn('geo $vn_website_ai_allowed {\n    default 0;', rendered)
        self.assertEqual(versions, {'main': 1, 'ai': 1, 'wiki': 1})
        data = self.data()
        data['policies'][1]['rules'][0][
            'network'
        ] = '1.1.1.1; include /etc/passwd;'
        with self.assertRaises(ValueError):
            render(data)
        data = self.data()
        data['policies'][1]['rules'] = []
        with self.assertRaises(ValueError):
            render(data)
        with self.assertRaises(ValueError):
            render({'policies': []})

    def test_reload_failure_restores_last_good_without_success_receipt(self):
        with TemporaryDirectory() as d:
            target = Path(d) / 'geo.conf'
            good = Path(d) / 'last-good.conf'
            receipt = Path(d) / 'status.json'
            target.write_text('old\n')
            good.write_text('old\n')
            config = {
                'reader': ['reader'],
                'hosts': self.data()['hosts'],
                'output': str(target),
                'last_good': str(good),
                'status': str(receipt),
            }
            with patch('deploy.website_access.agent.subprocess.run') as run:
                run.side_effect = [
                    SimpleNamespace(stdout=json.dumps(self.data())),
                    None,
                    RuntimeError('reload failed'),
                    None,
                ]
                with self.assertRaises(RuntimeError):
                    apply_once(config)
            self.assertEqual(target.read_text(), 'old\n')
            self.assertFalse(receipt.exists())

    def test_restart_after_interrupted_apply_reloads_before_acknowledging(
        self,
    ):
        with TemporaryDirectory() as d:
            target = Path(d) / 'geo.conf'
            good = Path(d) / 'last-good.conf'
            receipt = Path(d) / 'status.json'
            rendered, versions = render(self.data())
            target.write_text(rendered)  # Crash happened before reload.
            good.write_text('old\n')
            config = {
                'reader': ['reader'],
                'hosts': self.data()['hosts'],
                'output': str(target),
                'last_good': str(good),
                'status': str(receipt),
            }
            with patch('deploy.website_access.agent.subprocess.run') as run:
                run.side_effect = [
                    SimpleNamespace(stdout=json.dumps(self.data())),
                    None,
                    None,
                ]
                apply_once(config)
                self.assertEqual(run.call_count, 3)
            self.assertEqual(good.read_text(), rendered)
            self.assertEqual(
                json.loads(receipt.read_text())['versions'],
                versions,
            )

    def test_runtime_patches_are_idempotent_and_preserve_existing_edits(self):
        content = 'custom\nanchor\n'
        once = patched_code(content, [('anchor', 'anchor\naddition')])
        self.assertEqual(
            patched_code(once, [('anchor', 'anchor\naddition')]),
            once,
        )
        self.assertTrue(once.startswith('custom'))
        with self.assertRaises(ValueError):
            patched_code('different', [('anchor', 'addition')])
        gateway = (
            'server {\n server_name ai.changdar-server.mooo.com;\n'
            ' ssl_certificate /cert;\n'
            ' location / { auth_request /existing; }\n}'
        )
        updated = gateway_content(gateway, 'ai')
        self.assertIn('auth_request /existing', updated)
        self.assertEqual(gateway_content(updated, 'ai'), updated)
