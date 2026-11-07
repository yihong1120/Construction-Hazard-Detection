from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import auto_exit as a
import configure_exit as x


class AutoExitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for owner, name, value in [
            (a.d, 'ROOT', root),
            (a, 'PENDING', root / 'pending.json'),
            (a, 'RECEIPT', root / 'receipt.json'),
            (x, 'STATE', root / 'state.json'),
        ]:
            p = patch.object(owner, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.state = {
            'complete': True,
            'server_peer_id': 'server',
            'client_group_id': 'manual',
            'exit_group_id': 'exits',
            'client_peer_ids': ['mac'],
            'client_ips': ['10.203.252.213'],
            'policy_id': 'policy',
            'dns_id': 'dns',
            'route_id': 'route',
        }
        a.d.save(x.STATE, self.state)
        self.groups = [
            {
                'id': 'admins',
                'name': 'vpn-admins',
                'issued': 'jwt',
                'peers': [{'id': 'mac'}, {'id': 'iphone'}],
            },
            {'id': 'all', 'name': 'All', 'issued': 'api', 'peers': []},
        ]
        self.users = [
            {
                'id': 'employee',
                'idp_id': 'company',
                'is_service_user': False,
                'pending_approval': False,
                'is_blocked': False,
                'auto_groups': ['admins'],
            },
        ]
        self.peers = [
            {'id': 'mac', 'user_id': 'employee'},
            {'id': 'iphone', 'user_id': 'employee'},
            {'id': 'server', 'user_id': ''},
        ]
        self.settings = {
            'network_range': a.d.VPN_RANGE,
            'jwt_groups_enabled': True,
            'jwt_groups_claim_name': 'groups',
            'jwt_allow_groups': a.s.GROUPS,
            'groups_propagation_enabled': True,
            'peer_expose_enabled': False,
            'extra': {
                'user_approval_required': True,
                'peer_approval_enabled': False,
            },
        }
        values = x.payloads(self.state, True)
        self.objects = {
            kind + '/' + self.state[key]: copy.deepcopy(values[kind])
            for kind, key in a.OBJECTS
        }
        self.writes = []
        self.fail = None
        p = patch.object(a.d, 'api', side_effect=self.api)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(a.s, 'provider_for', return_value={'id': 'company'})
        p.start()
        self.addCleanup(p.stop)

    def api(self, path, method='GET', payload=None, token=None):
        if method != 'GET':
            self.writes.append((path, method, copy.deepcopy(payload)))
            if self.fail == path:
                self.fail = None
                raise RuntimeError('simulated API failure')
            if path == 'accounts/account':
                self.settings = copy.deepcopy(payload['settings'])
            elif path in self.objects:
                self.objects[path] = copy.deepcopy(payload)
            elif path.endswith('/approve'):
                user = next(
                    u for u in self.users if u['id'] == path.split('/')[1]
                )
                user.update(pending_approval=False, is_blocked=False)
            else:
                raise AssertionError('Unexpected mutation: ' + path)
            return {}
        if path == 'accounts':
            return [
                {'id': 'account', 'settings': copy.deepcopy(self.settings)},
            ]
        if path == 'groups':
            return copy.deepcopy(self.groups)
        if path.startswith('groups/'):
            return copy.deepcopy(
                next(g for g in self.groups if g['id'] == path.split('/')[1]),
            )
        if path == 'users':
            return copy.deepcopy(self.users)
        if path == 'peers':
            return copy.deepcopy(self.peers)
        return copy.deepcopy(self.objects[path])

    def enable(self, timer_error=None):
        with patch.object(x, 'verify'), patch.object(
            a,
            'install_timer',
            side_effect=timer_error,
        ):
            a.enable('fixture')

    def test_enable_uses_native_oidc_groups_for_route_dns_and_policy(self):
        self.enable()
        self.assertEqual(self.objects['routes/route']['groups'], ['admins'])
        self.assertEqual(
            self.objects['dns/nameservers/dns']['groups'],
            ['admins'],
        )
        self.assertEqual(
            self.objects['policies/policy']['rules'][0]['sources'],
            ['admins'],
        )
        self.assertFalse(self.settings['extra']['user_approval_required'])
        self.assertEqual(self.settings['jwt_allow_groups'], a.s.GROUPS)
        self.assertEqual(
            json.loads(x.STATE.read_text())['distribution_mode'],
            'oidc',
        )
        self.assertTrue(
            Path(json.loads(a.RECEIPT.read_text())['backup']).exists(),
        )
        self.assertFalse(
            any(
                path.startswith(('groups/', 'users/'))
                for path, _, _ in self.writes
            ),
        )

    def test_new_employee_group_is_distributed_without_manual_peer_updates(
        self,
    ):
        self.enable()
        self.groups.append(
            {
                'id': 'employees',
                'name': 'vpn-employees',
                'issued': 'jwt',
                'peers': [],
            },
        )
        self.writes.clear()
        a.sync('fixture')
        self.assertEqual(
            self.objects['routes/route']['groups'],
            ['admins', 'employees'],
        )
        self.assertEqual(
            self.objects['dns/nameservers/dns']['groups'],
            ['admins', 'employees'],
        )
        self.assertEqual(len(self.writes), 3)
        self.writes.clear()
        a.sync('fixture')
        self.assertEqual(self.writes, [])

    def test_manual_or_duplicate_claim_group_is_rejected(self):
        for variation in ('manual', 'duplicate'):
            with self.subTest(variation=variation):
                original = copy.deepcopy(self.groups)
                if variation == 'manual':
                    self.groups[0]['issued'] = 'api'
                else:
                    self.groups.append(copy.deepcopy(self.groups[0]))
                with self.assertRaises(RuntimeError):
                    a.resolve_groups('fixture')
                self.groups = original

    def test_missing_groups_does_not_fall_back_to_all(self):
        self.groups.pop(0)
        with self.assertRaises(RuntimeError):
            a.resolve_groups('fixture')

    def test_other_provider_and_setup_key_devices_are_rejected(self):
        for change in ('provider', 'server'):
            with self.subTest(change=change):
                if change == 'provider':
                    self.users[0]['idp_id'] = 'other'
                else:
                    self.users[0]['idp_id'] = 'company'
                    self.groups[0]['peers'].append({'id': 'server'})
                with self.assertRaises(RuntimeError):
                    a.validate_members(
                        'fixture',
                        {'vpn-admins': 'admins'},
                        'server',
                    )

    def test_api_failure_restores_all_objects_and_original_state(self):
        original = copy.deepcopy(self.objects)
        self.fail = 'dns/nameservers/dns'
        after = {
            **self.state,
            'distribution_mode': 'oidc',
            'oidc_distribution_groups': {'vpn-admins': 'admins'},
        }
        with self.assertRaises(RuntimeError):
            a.transition('fixture', self.state, after)
        self.assertEqual(self.objects, original)
        self.assertEqual(json.loads(x.STATE.read_text()), self.state)
        self.assertFalse(a.PENDING.exists())

    def test_killed_transition_resumes_partially_updated_objects(self):
        after = {
            **self.state,
            'distribution_mode': 'oidc',
            'oidc_distribution_groups': {'vpn-admins': 'admins'},
        }
        a.d.save(a.PENDING, {'before': self.state, 'after': after})
        self.objects['policies/policy'] = x.payloads(after, True)['policies']
        a.resume('fixture')
        self.assertEqual(json.loads(x.STATE.read_text()), after)
        self.assertFalse(a.PENDING.exists())
        self.assertEqual(len(self.writes), 2)

    def test_independent_object_edits_are_not_overwritten(self):
        self.objects['routes/route']['network'] = '192.168.1.0/24'
        after = {
            **self.state,
            'distribution_mode': 'oidc',
            'oidc_distribution_groups': {'vpn-admins': 'admins'},
        }
        with self.assertRaises(RuntimeError):
            a.transition('fixture', self.state, after)
        self.assertEqual(self.writes, [])

    def test_failed_timer_install_restores_original_approval_and_scope(self):
        original = copy.deepcopy(self.objects)
        with patch.object(a.subprocess, 'run'), self.assertRaises(
            RuntimeError,
        ):
            self.enable(RuntimeError('timer install failed'))
        self.assertEqual(self.objects, original)
        self.assertTrue(self.settings['extra']['user_approval_required'])
        self.assertEqual(json.loads(x.STATE.read_text()), self.state)
        # The same command can retry the prepared migration.
        self.enable()
        self.assertEqual(
            json.loads(a.RECEIPT.read_text())['phase'],
            'complete',
        )

    def test_pending_company_identities_exclude_blocked_users(
        self,
    ):
        self.users[0].update(pending_approval=True, is_blocked=True)
        self.users.extend(
            [
                {
                    'id': 'foreign',
                    'idp_id': 'other',
                    'pending_approval': True,
                    'is_blocked': True,
                    'auto_groups': ['admins'],
                },
                {
                    'id': 'blocked',
                    'idp_id': 'company',
                    'pending_approval': False,
                    'is_blocked': True,
                    'auto_groups': ['admins'],
                },
                {
                    'id': 'unverified',
                    'idp_id': 'company',
                    'pending_approval': True,
                    'is_blocked': True,
                    'auto_groups': [],
                },
            ],
        )
        a.approve_pending('fixture', {'vpn-admins': 'admins'})
        self.assertEqual(
            [w[0] for w in self.writes],
            ['users/employee/approve'],
        )


if __name__ == '__main__':
    unittest.main()
